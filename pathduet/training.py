import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score
from .model import PathDUET
from .losses import compute_disentanglement_losses
from .runtime import make_loader, clone_state_dict, collect_probs_and_labels


def make_pos_weight(y_train: np.ndarray, device) -> torch.Tensor:
    class_counts = np.bincount(np.asarray(y_train).astype(int), minlength=2)
    pos_weight = class_counts[0] / max(class_counts[1], 1)
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


def train_one_fold(data, gene_path_mask_df, pathway_semantic_memory, device, output):
    train_arrays, val_arrays, test_arrays = [data[part]["arrays"] for part in ("train", "validation", "test")]
    y_train, y_val, y_test = [data[part]["labels"] for part in ("train", "validation", "test")]
    train_loader = make_loader(train_arrays, y_train, 32, True)
    val_loader = make_loader(val_arrays, y_val, 32, False)
    test_loader = make_loader(test_arrays, y_test, 32, False)
    model = PathDUET(
        n_genes=len(gene_path_mask_df),
        gene_path_mask_df=gene_path_mask_df,
        pathway_semantic_memory=pathway_semantic_memory,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0005, weight_decay=0.0001
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=make_pos_weight(y_train, device))
    best_val_auc = float("-inf")
    best_state = None
    patience_counter = 0
    for _epoch in range(120):
        if hasattr(model, "set_fusion_progress"):
            model.set_fusion_progress(_epoch / max(120 - 1, 1))
        model.train()
        for bx_m, bx_mi, bx_d, bx_c, by in train_loader:
            by_device = by.to(device)
            outputs = model(
                bx_m.to(device),
                bx_mi.to(device),
                bx_d.to(device),
                bx_c.to(device),
                targets=by_device,
            )
            losses = compute_disentanglement_losses(outputs, by_device, criterion)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        (val_true, val_probs) = collect_probs_and_labels(model, val_loader)
        val_auc = roc_auc_score(val_true, val_probs)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = clone_state_dict(model)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break
    if best_state is None:
        raise RuntimeError("Training finished without storing a best model state.")
    model.load_state_dict(best_state)
    test_true, test_probs = collect_probs_and_labels(model, test_loader)
    auc = float(roc_auc_score(test_true, test_probs))
    aupr = float(average_precision_score(test_true, test_probs))
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output / "model.pt")
    pd.DataFrame({
        "sample_id": data["test"]["ids"],
        "label": test_true,
        "probability": test_probs,
    }).to_csv(output / "predictions.csv", index=False)
    return {"auc": auc, "aupr": aupr}
