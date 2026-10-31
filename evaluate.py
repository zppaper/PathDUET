from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from pathduet.data import fold_digest, load_fold, load_resources, resource_digest
from predict import load_checkpoint, predict_arrays
from train import COHORTS, SEEDS, sha

ROOT = Path(__file__).resolve().parent


def evaluate(results, data_root, output, require_complete=False, replay=False, device="cpu"):
    rows = []
    errors = []
    expected = {
        ("stage", cohort, seed, fold)
        for cohort in COHORTS
        for seed in SEEDS
        for fold in range(1, 6)
    }
    observed = set()
    identities = {}
    for receipt in sorted(results.glob("*/*/seed_*/fold_*/complete.json")):
        directory = receipt.parent
        task, cohort = directory.parents[2].name, directory.parents[1].name
        seed = int(directory.parent.name.split("_")[1])
        fold = int(directory.name.split("_")[1])
        key = task, cohort, seed, fold
        try:
            if key not in expected:
                raise ValueError(f"Unexpected fold identity: {key}")
            metadata = json.loads((directory.parent / "run.json").read_text())
            if (metadata["task"], metadata["cohort"], metadata["seed"]) != key[:3]:
                raise ValueError("Run identity differs from its stage result directory.")
            if metadata["smoke"]:
                raise ValueError("Smoke runs cannot contribute to reproduction statistics.")
            record = json.loads(receipt.read_text())
            for name, digest in record["artifacts"].items():
                if sha(directory / name) != digest:
                    raise ValueError(f"Artifact hash mismatch: {name}")
            fold_file = data_root / "stage" / cohort / "splits" / f"seed_{seed}" / f"fold_{fold}.csv"
            if fold_digest(fold_file) != record["input_sha256"]:
                raise ValueError("Prepared input hash mismatch.")
            resources = data_root / "resources"
            if resource_digest(resources) != metadata["resources_sha256"]:
                raise ValueError("CSV resources differ from the training configuration.")
            mask, _ = load_resources(resources)
            data = load_fold(fold_file, mask.index)
            predictions = pd.read_csv(directory / "predictions.csv")
            if predictions.sample_id.astype(str).tolist() != data["test"]["ids"].tolist():
                raise ValueError("Prediction sample order differs from held-out data.")
            np.testing.assert_array_equal(predictions.label.to_numpy(), data["test"]["labels"])
            if (
                not np.isfinite(predictions.probability).all()
                or not predictions.probability.between(0, 1).all()
            ):
                raise ValueError("Invalid predicted probabilities.")
            for sample in data["test"]["ids"]:
                patient_key = task, cohort, seed, sample
                if patient_key in identities:
                    raise ValueError("A patient appears in more than one outer test fold.")
                identities[patient_key] = fold
            auc = float(roc_auc_score(predictions.label, predictions.probability))
            ap = float(average_precision_score(predictions.label, predictions.probability))
            metrics = json.loads((directory / "metrics.json").read_text())
            np.testing.assert_allclose(
                [auc, ap], [metrics["auc"], metrics["aupr"]], atol=1e-12, rtol=0
            )
            error = None
            if replay:
                model, saved = load_checkpoint(directory / "checkpoint.pt", resources, device)
                if (saved["task"], saved["cohort"], saved["seed"], saved["fold"]) != key:
                    raise ValueError("Checkpoint identity differs from its fold.")
                probabilities = predict_arrays(model, data["test"]["arrays"])
                error = float(np.max(np.abs(probabilities - predictions.probability.to_numpy())))
                if error > 2e-6:
                    raise ValueError(f"Checkpoint replay differs by {error}.")
                del model
            rows.append(
                {
                    "task": task,
                    "cohort": cohort,
                    "seed": seed,
                    "fold": fold,
                    "auc": auc,
                    "aupr": ap,
                    "checkpoint_max_error": error,
                }
            )
            observed.add(key)
        except Exception as exc:
            errors.append({"fold": list(key), "error": str(exc)})
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.to_csv(output / "fold_metrics.csv", index=False)
        counts = frame.groupby(["task", "cohort", "seed"]).size()
        complete_keys = counts[counts == 5].index
        seed_rows = (
            frame.groupby(["task", "cohort", "seed"])[["auc", "aupr"]]
            .mean()
            .loc[complete_keys]
            .reset_index()
        )
        seed_rows.to_csv(output / "seed_metrics.csv", index=False)
        complete_cohorts = seed_rows.groupby(["task", "cohort"]).size()
        accepted = set(complete_cohorts[complete_cohorts == 3].index)
        selected = seed_rows[
            [tuple(row) in accepted for row in seed_rows[["task", "cohort"]].to_numpy()]
        ]
        if len(selected):
            summary = selected.groupby(["task", "cohort"])[["auc", "aupr"]].agg(["mean", "std"])
            summary.to_csv(output / "cohort_mean_sd.csv")
        if len(seed_rows) == len(COHORTS) * len(SEEDS):
            macro = seed_rows.groupby("seed")[["auc", "aupr"]].mean()
            macro.to_csv(output / "macro_seed_metrics.csv")
            macro.agg(["mean", "std"]).to_csv(output / "macro_mean_sd.csv")
    report = {
        "complete": observed == expected and not errors,
        "verified_folds": len(observed),
        "expected_folds": len(expected),
        "replayed": replay,
        "missing": sorted(expected - observed),
        "errors": errors,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"complete": report["complete"], "verified_folds": len(observed), "errors": len(errors)}
        )
    )
    if errors or (require_complete and not report["complete"]):
        raise RuntimeError("Evaluation is incomplete or failed; inspect verification.json.")


def main():
    parser = argparse.ArgumentParser(
        description="Verify held-out predictions and aggregate complete seed runs."
    )
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.results, args.data, args.output, args.require_complete, args.replay, args.device)


if __name__ == "__main__":
    main()
