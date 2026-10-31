from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import torch

from pathduet.data import fold_digest, load_fold, load_resources, resource_digest
from pathduet.runtime import capture_rng_state, restore_rng_state, set_seeds
from pathduet.training import train_one_fold

ROOT = Path(__file__).resolve().parent
COHORTS = ("COAD", "KIRP", "THCA", "UCS")
SEEDS = (1028, 2024, 3407)
STAGE_PARAMETERS = {'SEMANTIC_MEMORY_TEMPERATURE': 0.5,
 'SEMANTIC_MEMORY_INIT_SCALE': 0.01,
 'SEMANTIC_MEMORY_HIDDEN_DIM': 128,
 'BATCH_SIZE': 32,
 'LEARNING_RATE': 0.0005,
 'WEIGHT_DECAY': 0.0001,
 'EPOCHS': 120,
 'PATIENCE': 20,
 'GRAD_CLIP_NORM': 5.0,
 'SHARED_DIM': 128,
 'PATHWAY_HIDDEN_DIM': 192,
 'UNIQUE_DIM': 128,
 'UNIQUE_HIDDEN_DIM': 192,
 'CLASSIFIER_HIDDEN_DIM': 256,
 'DROPOUT': 0.3,
 'ENABLE_RESIDUAL_BALANCED_FUSION': True,
 'FUSION_WARMUP_RATIO': 0.149589,
 'FUSION_STRONG_SUPPRESSION_GAMMA': 0.05439,
 'FUSION_WEAK_BOOST_BETA': 0.137326,
 'FUSION_SCALE_MIN': 0.84543,
 'FUSION_SCALE_MAX': 1.170581,
 'FUSION_CONTRIBUTION_TARGET': 0.38663,
 'FUSION_WEAK_PARTICIPATION_FLOOR': 0.150732,
 'FUSION_BALANCE_LOSS_WEIGHT': 0.02,
 'FUSION_BALANCED_AUX_LOSS_WEIGHT': 0.0,
 'FUSION_BALANCED_CONSISTENCY_WEIGHT': 0.0,
 'FUSION_FIXED_BLEND_WEIGHT': 0.130923,
 'BRANCH_AUX_LOSS_WEIGHT': 0.002032,
 'ICG_BALANCE_LOSS_WEIGHT': 0.047331,
 'ICG_TARGET_GAP': 0.151118,
 'ICG_DROPOUT_BASE': 0.257131,
 'ICG_DROPOUT_MAX': 0.494363,
 'ICG_DROPOUT_WARMUP_RATIO': 0.258335,
 'ICG_SHARED_DROPOUT_SCALE': 0.113388,
 'ALIGN_LOSS_WEIGHT': 1.0,
 'SHARED_UNIQUE_ORTH_WEIGHT': 0.3,
 'UNIQUE_UNIQUE_ORTH_WEIGHT': 0.2,
 'RECON_LOSS_WEIGHT': 1.0,
 'PATHWAY_RECON_LOSS_WEIGHT': 0.5}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    task = "stage"
    selected = args.cohorts or list(COHORTS)
    if set(selected) - set(COHORTS) or len(selected) != len(set(selected)):
        raise ValueError("Choose unique stage cohorts from COAD, KIRP, THCA and UCS.")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be unique.")
    seeds = args.seeds
    source_hashes = {
        str(p.relative_to(ROOT)): sha(p)
        for p in [Path(__file__)] + sorted((ROOT / "pathduet").glob("*.py"))
    }
    config = dict(STAGE_PARAMETERS)
    if args.smoke:
        config.update(EPOCHS=2, PATIENCE=2)
    resource_path = args.data / "resources"
    mask, memory = load_resources(resource_path)
    for cohort in selected:
        for seed in seeds:
            seed_dir = args.output / task / cohort / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "task": task,
                "cohort": cohort,
                "seed": seed,
                "configuration": config,
                "source_hashes": source_hashes,
                "resources_sha256": resource_digest(resource_path),
                "input_format": "csv_v1",
                "smoke": args.smoke,
                "device": str(device),
                "threads": args.threads,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            }
            config_path = seed_dir / "run.json"
            if config_path.exists() and json.loads(config_path.read_text()) != metadata:
                raise RuntimeError(
                    f"Existing output belongs to another configuration: {seed_dir}"
                )
            dump(config_path, metadata)
            set_seeds(seed)
            for fold in range(1, 2 if args.smoke else 6):
                fold_path = args.data / "stage" / cohort / "splits" / f"seed_{seed}" / f"fold_{fold}.csv"
                digest = fold_digest(fold_path)
                directory = seed_dir / f"fold_{fold}"
                directory.mkdir(exist_ok=True)
                receipt = directory / "complete.json"
                if receipt.exists():
                    completed = json.loads(receipt.read_text())
                    if completed["input_sha256"] != digest:
                        raise RuntimeError("Prepared input changed after this fold completed.")
                    for name, expected in completed["artifacts"].items():
                        if sha(directory / name) != expected:
                            raise RuntimeError(f"Changed output artifact: {directory / name}")
                    restore_rng_state(
                        torch.load(directory / "rng.pt", weights_only=False, map_location="cpu")
                    )
                    print("RESUME", task, cohort, seed, fold, flush=True)
                    continue
                data = load_fold(fold_path, mask.index)
                started = time.time()

                def save_predictions(
                    model, state, labels, probabilities, epochs, validation_auc
                ):
                    pd.DataFrame(
                        {
                            "sample_id": data["test"]["ids"],
                            "label": labels,
                            "probability": probabilities,
                        }
                    ).to_csv(directory / "predictions.csv", index=False)
                    torch.save(
                        {
                            "state_dict": state,
                            "configuration": config,
                            "task": task,
                            "cohort": cohort,
                            "seed": seed,
                            "fold": fold,
                            "genes": mask.index.tolist(),
                            "pathways": mask.columns.tolist(),
                            "resources_sha256": metadata["resources_sha256"],
                            "scaler_sha256": sha(fold_path.with_name(fold_path.stem + "_scaler.csv")),
                        },
                        directory / "checkpoint.pt",
                    )
                    dump(
                        directory / "training.json",
                        {
                            "epochs_run": epochs,
                            "best_validation_auc": validation_auc,
                            "selected_progress": float(state["fusion_progress"]),
                            "seconds": time.time() - started,
                        },
                    )

                print("START", task, cohort, seed, fold, flush=True)
                metrics = train_one_fold(
                    *[data[k]["arrays"] for k in ("train", "validation", "test")],
                    *[data[k]["labels"] for k in ("train", "validation", "test")],
                    mask,
                    len(mask),
                    memory,
                    cohort,
                    seed,
                    fold,
                    config,
                    device,
                    save_predictions,
                )
                torch.save(capture_rng_state(), directory / "rng.pt")
                dump(directory / "metrics.json", metrics)
                artifacts = {
                    name: sha(directory / name)
                    for name in (
                        "checkpoint.pt",
                        "predictions.csv",
                        "metrics.json",
                        "rng.pt",
                        "training.json",
                    )
                }
                dump(receipt, {"input_sha256": digest, "artifacts": artifacts})
                print(
                    "DONE",
                    task,
                    cohort,
                    seed,
                    fold,
                    metrics["auc"],
                    metrics["aupr"],
                    round(time.time() - started, 1),
                    flush=True,
                )


def main():
    parser = argparse.ArgumentParser(description="Train PathDUET for cancer-stage classification with built-in stage parameters.")
    parser.add_argument("--cohorts", nargs="+", choices=COHORTS)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--data", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.output = args.output / "smoke"
    run(args)


if __name__ == "__main__":
    main()
