from pathlib import Path
import argparse
import hashlib

import numpy as np
import pandas as pd
import torch

from pathduet.data import load_fold, load_inputs, load_resources, resource_digest
from pathduet.model import PathDUET


def load_checkpoint(checkpoint, resources, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("task") != "stage":
        raise ValueError("This release accepts cancer-stage checkpoints only.")
    digest = resource_digest(resources)
    if digest != saved["resources_sha256"]:
        raise ValueError("Resources differ from those used to train this checkpoint.")
    mask, memory = load_resources(resources)
    if saved["genes"] != mask.index.tolist() or saved["pathways"] != mask.columns.tolist():
        raise ValueError("Checkpoint gene or pathway order does not match the supplied resources.")
    config = saved["configuration"]
    model = PathDUET(
        len(mask),
        mask,
        config["SHARED_DIM"],
        config["PATHWAY_HIDDEN_DIM"],
        config["UNIQUE_DIM"],
        config["UNIQUE_HIDDEN_DIM"],
        config["CLASSIFIER_HIDDEN_DIM"],
        config["DROPOUT"],
        memory,
        config=config,
    ).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model, saved


def predict_arrays(model, arrays, batch_size=32):
    device = next(model.parameters()).device
    if len(arrays) != 4 or len(arrays[0]) == 0:
        raise ValueError("Four nonempty prepared omics matrices are required.")
    shape = arrays[0].shape
    if any(a.shape != shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Input matrices must have equal shapes and finite values.")
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(arrays[0]), batch_size):
            batch = [
                torch.as_tensor(a[start : start + batch_size], dtype=torch.float32, device=device)
                for a in arrays
            ]
            probabilities.append(torch.sigmoid(model(*batch)["logits"]).cpu().numpy().ravel())
    return np.concatenate(probabilities)


def main():
    parser = argparse.ArgumentParser(description="Predict from prepared, model-compatible inputs.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--resources", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--scaler", type=Path)
    parser.add_argument(
        "--partition", choices=["train", "validation", "test", "input"], default="test"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, saved = load_checkpoint(args.checkpoint, args.resources, args.device)
    if args.partition == "input":
        if args.scaler is None:
            parser.error("--partition input requires --scaler from the checkpoint's training fold.")
        if hashlib.sha256(args.scaler.read_bytes()).hexdigest() != saved["scaler_sha256"]:
            raise ValueError("Scaler differs from the checkpoint's training fold.")
        arrays, ids = load_inputs(args.input, saved["genes"], args.scaler)
    else:
        scaler = args.input.with_name(args.input.stem + "_scaler.csv")
        if hashlib.sha256(scaler.read_bytes()).hexdigest() != saved["scaler_sha256"]:
            raise ValueError("Input fold scaler differs from the checkpoint's training fold.")
        data = load_fold(args.input, saved["genes"])[args.partition]
        arrays, ids = data["arrays"], data["ids"]
    if len(ids) != len(arrays[0]) or len(set(ids)) != len(ids):
        raise ValueError("Input sample identifiers are missing or duplicated.")
    probabilities = predict_arrays(model, arrays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": ids, "probability": probabilities}).to_csv(args.output, index=False)
    print(f"Saved {len(ids)} predictions to {args.output}")


if __name__ == "__main__":
    main()
