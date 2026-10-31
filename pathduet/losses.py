from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F
from .constants import MODALITY_NAMES, REPRESENTATION_NAMES


def cross_covariance_penalty(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    x = x / x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-06)
    y = y / y.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-06)
    denom = max(x.shape[0] - 1, 1)
    covariance = x.T @ y / denom
    return covariance.pow(2).mean()


def compute_disentanglement_losses(
    outputs: Dict[str, object], targets: torch.Tensor, criterion: nn.Module, config
) -> Dict[str, torch.Tensor]:
    classification_loss = criterion(outputs["logits"].squeeze(-1), targets)
    balanced_logits = outputs.get("balanced_logits")
    fusion_balance_info = outputs.get("fusion_balance", {})
    balance_loss_scale = fusion_balance_info.get("balance_loss_scale")
    if not isinstance(balance_loss_scale, torch.Tensor):
        balance_loss_scale = torch.ones((), device=targets.device, dtype=classification_loss.dtype)
    if isinstance(balanced_logits, torch.Tensor):
        balanced_classification_loss = criterion(balanced_logits.squeeze(-1), targets)
        consistency_loss = F.mse_loss(
            torch.sigmoid(balanced_logits.squeeze(-1)),
            torch.sigmoid(outputs["raw_logits"].detach().squeeze(-1)),
        )
        balanced_aux_loss = balance_loss_scale * (
            config["FUSION_BALANCED_AUX_LOSS_WEIGHT"] * balanced_classification_loss
            + config["FUSION_BALANCED_CONSISTENCY_WEIGHT"] * consistency_loss
        )
    else:
        balanced_classification_loss = torch.zeros(
            (), device=targets.device, dtype=classification_loss.dtype
        )
        consistency_loss = torch.zeros((), device=targets.device, dtype=classification_loss.dtype)
        balanced_aux_loss = torch.zeros((), device=targets.device, dtype=classification_loss.dtype)
    branch_logits = outputs.get("branch_logits", {})
    branch_losses = []
    branch_confidences = []
    if isinstance(branch_logits, dict):
        for name in REPRESENTATION_NAMES:
            branch_logit = branch_logits.get(name)
            if isinstance(branch_logit, torch.Tensor):
                branch_logit_flat = branch_logit.squeeze(-1)
                branch_losses.append(criterion(branch_logit_flat, targets))
                branch_prob = torch.sigmoid(branch_logit_flat)
                correct_confidence = targets * branch_prob + (1.0 - targets) * (1.0 - branch_prob)
                branch_confidences.append(correct_confidence)
    if branch_losses:
        branch_aux_loss = torch.stack(branch_losses).mean()
    else:
        branch_aux_loss = torch.zeros((), device=targets.device, dtype=classification_loss.dtype)
    if branch_confidences:
        confidence_matrix = torch.stack(branch_confidences, dim=1)
        icg_per_sample = confidence_matrix.max(dim=1).values - confidence_matrix.min(dim=1).values
        icg_balance_loss = F.relu(icg_per_sample - config["ICG_TARGET_GAP"]).pow(2).mean()
        icg_value = icg_per_sample.mean()
    else:
        icg_balance_loss = torch.zeros((), device=targets.device, dtype=classification_loss.dtype)
        icg_value = torch.zeros((), device=targets.device, dtype=classification_loss.dtype)
    align_loss = torch.stack(
        [
            F.mse_loss(outputs["shared_candidates"][name], outputs["z_shared"])
            for name in MODALITY_NAMES
        ]
    ).mean()
    shared_unique_loss = torch.stack(
        [
            cross_covariance_penalty(outputs["z_shared"], outputs["z_unique"][name])
            for name in MODALITY_NAMES
        ]
    ).mean()
    unique_unique_terms = []
    for idx, left_name in enumerate(MODALITY_NAMES):
        for right_name in MODALITY_NAMES[idx + 1 :]:
            unique_unique_terms.append(
                cross_covariance_penalty(
                    outputs["z_unique"][left_name], outputs["z_unique"][right_name]
                )
            )
    unique_unique_loss = torch.stack(unique_unique_terms).mean()
    recon_loss = torch.stack(
        [
            F.mse_loss(outputs["recon_gene"][name], outputs["inputs"][name])
            for name in MODALITY_NAMES
        ]
    ).mean()
    pathway_recon_loss = torch.stack(
        [
            F.mse_loss(outputs["shared_pathway"][name], outputs["pathway_inputs"][name])
            for name in MODALITY_NAMES
        ]
    ).mean()
    fusion_concentration = fusion_balance_info.get("concentration")
    if isinstance(fusion_concentration, torch.Tensor):
        concentration_target = max(
            config["FUSION_CONTRIBUTION_TARGET"], 1.0 / len(REPRESENTATION_NAMES)
        )
        concentration_loss = F.relu(fusion_concentration - concentration_target).pow(2).mean()
        weak_participation_loss = fusion_balance_info.get("weak_participation_loss")
        if not isinstance(weak_participation_loss, torch.Tensor):
            weak_participation_loss = torch.zeros(
                (), device=targets.device, dtype=classification_loss.dtype
            )
        fusion_balance_loss = balance_loss_scale * (concentration_loss + weak_participation_loss)
    else:
        fusion_balance_loss = torch.zeros(
            (), device=targets.device, dtype=classification_loss.dtype
        )
    total_loss = (
        classification_loss
        + config["ALIGN_LOSS_WEIGHT"] * align_loss
        + config["SHARED_UNIQUE_ORTH_WEIGHT"] * shared_unique_loss
        + config["UNIQUE_UNIQUE_ORTH_WEIGHT"] * unique_unique_loss
        + config["RECON_LOSS_WEIGHT"] * recon_loss
        + config["PATHWAY_RECON_LOSS_WEIGHT"] * pathway_recon_loss
        + config["FUSION_BALANCE_LOSS_WEIGHT"] * fusion_balance_loss
        + balanced_aux_loss
        + config["BRANCH_AUX_LOSS_WEIGHT"] * branch_aux_loss
        + config["ICG_BALANCE_LOSS_WEIGHT"] * icg_balance_loss
    )
    return {
        "total": total_loss,
        "classification": classification_loss,
        "balanced_classification": balanced_classification_loss,
        "fusion_consistency": consistency_loss,
        "balanced_aux": balanced_aux_loss,
        "branch_aux": branch_aux_loss,
        "icg_balance": icg_balance_loss,
        "icg_train": icg_value.detach(),
        "align": align_loss,
        "shared_unique": shared_unique_loss,
        "unique_unique": unique_unique_loss,
        "reconstruction": recon_loss,
        "pathway_reconstruction": pathway_recon_loss,
        "fusion_balance": fusion_balance_loss,
    }
