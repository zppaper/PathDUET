from typing import Dict, Sequence, Tuple
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .constants import REPRESENTATION_NAMES
from .metrics import (
    compute_fusion_imbalance_from_contributions,
    compute_icg_from_branch_confidences,
)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(
    arrays: Sequence[np.ndarray], y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(arrays[0], dtype=torch.float32),
        torch.tensor(arrays[1], dtype=torch.float32),
        torch.tensor(arrays[2], dtype=torch.float32),
        torch.tensor(arrays[3], dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for (name, tensor) in model.state_dict().items()}


def capture_rng_state() -> Dict[str, object]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def collect_probs_and_labels(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    probs = []
    labels = []
    model.eval()
    with torch.no_grad():
        for bx_m, bx_mi, bx_d, bx_c, by in loader:
            outputs = model(
                bx_m.to(next(model.parameters()).device),
                bx_mi.to(next(model.parameters()).device),
                bx_d.to(next(model.parameters()).device),
                bx_c.to(next(model.parameters()).device),
            )
            batch_probs = torch.sigmoid(outputs["logits"].squeeze(-1)).cpu().numpy()
            probs.extend(np.atleast_1d(batch_probs).tolist())
            labels.extend(by.numpy().tolist())
    return (np.asarray(labels).astype(int), np.asarray(probs, dtype=np.float64))


def collect_probs_labels_and_fusion_stats(
    model: nn.Module, loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    probs = []
    labels = []
    raw_contributions = []
    final_contributions = []
    balanced_contributions = []
    branch_confidences = {name: [] for name in REPRESENTATION_NAMES}
    model.eval()
    with torch.no_grad():
        for bx_m, bx_mi, bx_d, bx_c, by in loader:
            batch_targets = by.to(next(model.parameters()).device)
            outputs = model(
                bx_m.to(next(model.parameters()).device),
                bx_mi.to(next(model.parameters()).device),
                bx_d.to(next(model.parameters()).device),
                bx_c.to(next(model.parameters()).device),
            )
            batch_probs = torch.sigmoid(outputs["logits"].squeeze(-1)).cpu().numpy()
            probs.extend(np.atleast_1d(batch_probs).tolist())
            labels.extend(by.numpy().tolist())
            fusion_balance = outputs.get("fusion_balance", {})
            raw_contribution = fusion_balance.get("raw_contribution")
            final_contribution = fusion_balance.get("contribution")
            balanced_contribution = fusion_balance.get("balanced_contribution")
            if isinstance(raw_contribution, torch.Tensor):
                raw_contributions.append(raw_contribution.detach().cpu().numpy())
            if isinstance(final_contribution, torch.Tensor):
                final_contributions.append(final_contribution.detach().cpu().numpy())
            if isinstance(balanced_contribution, torch.Tensor):
                balanced_contributions.append(balanced_contribution.detach().cpu().numpy())
            branch_logits = outputs.get("branch_logits", {})
            if isinstance(branch_logits, dict):
                for name in REPRESENTATION_NAMES:
                    branch_logit = branch_logits.get(name)
                    if isinstance(branch_logit, torch.Tensor):
                        branch_prob = torch.sigmoid(branch_logit.squeeze(-1))
                        correct_confidence = batch_targets * branch_prob + (1.0 - batch_targets) * (
                            1.0 - branch_prob
                        )
                        branch_confidences[name].extend(
                            correct_confidence.detach().cpu().numpy().tolist()
                        )
    stats: Dict[str, float] = {}
    contribution_groups = {
        "raw": raw_contributions,
        "final": final_contributions,
        "balanced": balanced_contributions,
    }
    for prefix, chunks in contribution_groups.items():
        if chunks:
            group_stats = compute_fusion_imbalance_from_contributions(
                np.concatenate(chunks, axis=0)
            )
        else:
            group_stats = compute_fusion_imbalance_from_contributions(
                np.empty((0, len(REPRESENTATION_NAMES)))
            )
        stats.update({f"{prefix}_{name}": value for (name, value) in group_stats.items()})
    stats.update(compute_icg_from_branch_confidences(branch_confidences))
    return (np.asarray(labels).astype(int), np.asarray(probs, dtype=np.float64), stats)
