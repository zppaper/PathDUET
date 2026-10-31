from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from .model import PathDUET
from .losses import compute_disentanglement_losses
from .metrics import compute_metrics
from .runtime import (
    make_loader,
    clone_state_dict,
    collect_probs_and_labels,
    collect_probs_labels_and_fusion_stats,
)


def make_pos_weight(y_train: np.ndarray, device) -> torch.Tensor:
    class_counts = np.bincount(np.asarray(y_train).astype(int), minlength=2)
    pos_weight = class_counts[0] / max(class_counts[1], 1)
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


def train_one_fold(
    train_arrays: Sequence[np.ndarray],
    val_arrays: Sequence[np.ndarray],
    test_arrays: Sequence[np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    gene_path_mask_df: pd.DataFrame,
    n_genes: int,
    pathway_semantic_memory: Optional[np.ndarray],
    dataset: str,
    seed: int,
    fold_idx: int,
    config,
    device,
    save_predictions,
):
    train_loader = make_loader(train_arrays, y_train, config["BATCH_SIZE"], True)
    val_loader = make_loader(val_arrays, y_val, config["BATCH_SIZE"], False)
    test_loader = make_loader(test_arrays, y_test, config["BATCH_SIZE"], False)
    model = PathDUET(
        n_genes=n_genes,
        gene_path_mask_df=gene_path_mask_df,
        shared_dim=config["SHARED_DIM"],
        pathway_hidden_dim=config["PATHWAY_HIDDEN_DIM"],
        unique_dim=config["UNIQUE_DIM"],
        unique_hidden_dim=config["UNIQUE_HIDDEN_DIM"],
        classifier_hidden_dim=config["CLASSIFIER_HIDDEN_DIM"],
        dropout=config["DROPOUT"],
        pathway_semantic_memory=pathway_semantic_memory,
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"]
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=make_pos_weight(y_train, device))
    best_val_auc = float("-inf")
    best_state = None
    patience_counter = 0
    for _epoch in range(config["EPOCHS"]):
        if hasattr(model, "set_fusion_progress"):
            model.set_fusion_progress(_epoch / max(config["EPOCHS"] - 1, 1))
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
            losses = compute_disentanglement_losses(outputs, by_device, criterion, config)
            optimizer.zero_grad()
            losses["total"].backward()
            if config["GRAD_CLIP_NORM"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["GRAD_CLIP_NORM"])
            optimizer.step()
        (val_true, val_probs) = collect_probs_and_labels(model, val_loader)
        val_auc = roc_auc_score(val_true, val_probs)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = clone_state_dict(model)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["PATIENCE"]:
                break
    if best_state is None:
        raise RuntimeError("Training finished without storing a best model state.")
    model.load_state_dict(best_state)
    (test_true, test_probs, fusion_stats) = collect_probs_labels_and_fusion_stats(
        model, test_loader
    )
    (auc, aupr) = compute_metrics(test_true, test_probs)
    save_predictions(model, best_state, test_true, test_probs, _epoch + 1, best_val_auc)
    fold_result = {"auc": auc, "aupr": aupr, **fusion_stats}
    return fold_result
