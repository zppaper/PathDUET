from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .constants import MODALITY_NAMES, REPRESENTATION_NAMES


class ResidualMLPBlock(nn.Module):

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


def build_encoder(
    input_dim: int, hidden_dim: int, output_dim: int, dropout: float
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        ResidualMLPBlock(hidden_dim, dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.LayerNorm(output_dim),
    )


def build_decoder(
    input_dim: int, hidden_dim: int, output_dim: int, dropout: float
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        ResidualMLPBlock(hidden_dim, dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class PathDUET(nn.Module):

    def __init__(
        self,
        n_genes: int,
        gene_path_mask_df: pd.DataFrame,
        shared_dim: int,
        pathway_hidden_dim: int,
        unique_dim: int,
        unique_hidden_dim: int,
        classifier_hidden_dim: int,
        dropout: float,
        pathway_semantic_memory: Optional[np.ndarray] = None,
        *,
        config,
    ):
        super().__init__()
        self.config = dict(config)
        gene_path_binary = torch.tensor(gene_path_mask_df.values, dtype=torch.float32)
        gene_to_path = gene_path_binary / gene_path_binary.sum(dim=0, keepdim=True).clamp_min(1.0)
        path_to_gene = gene_path_binary.T / gene_path_binary.T.sum(dim=0, keepdim=True).clamp_min(
            1.0
        )
        self.register_buffer("gene_to_path", gene_to_path)
        self.register_buffer("path_to_gene", path_to_gene)
        n_pathways = gene_path_mask_df.shape[1]
        self.shared_encoder = build_encoder(n_pathways, pathway_hidden_dim, shared_dim, dropout)
        self.use_pathway_semantic_memory = pathway_semantic_memory is not None
        if self.use_pathway_semantic_memory:
            semantic_tensor = torch.tensor(pathway_semantic_memory, dtype=torch.float32)
            if semantic_tensor.ndim != 2 or semantic_tensor.shape[0] != n_pathways:
                raise ValueError(
                    f"pathway_semantic_memory must have shape [n_pathways, semantic_dim], got {tuple(semantic_tensor.shape)} for n_pathways={n_pathways}"
                )
            semantic_dim = int(semantic_tensor.shape[1])
            self.register_buffer("pathway_semantic_memory", semantic_tensor)
            self.semantic_temperature = max(
                float(self.config["SEMANTIC_MEMORY_TEMPERATURE"]), 1e-06
            )
            self.semantic_encoder = nn.Sequential(
                nn.Linear(semantic_dim, self.config["SEMANTIC_MEMORY_HIDDEN_DIM"]),
                nn.LayerNorm(self.config["SEMANTIC_MEMORY_HIDDEN_DIM"]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.config["SEMANTIC_MEMORY_HIDDEN_DIM"], shared_dim),
                nn.LayerNorm(shared_dim),
            )
            self.semantic_fusion_norm = nn.LayerNorm(shared_dim)
            self.semantic_residual_scale = nn.Parameter(
                torch.tensor(float(self.config["SEMANTIC_MEMORY_INIT_SCALE"]), dtype=torch.float32)
            )
        else:
            self.semantic_temperature = 1.0
            self.semantic_encoder = None
            self.semantic_fusion_norm = None
            self.semantic_residual_scale = None
        self.shared_pathway_decoders = nn.ModuleDict(
            {
                name: build_decoder(shared_dim, pathway_hidden_dim, n_pathways, dropout)
                for name in MODALITY_NAMES
            }
        )
        self.unique_encoders = nn.ModuleDict(
            {
                name: build_encoder(n_genes, unique_hidden_dim, unique_dim, dropout)
                for name in MODALITY_NAMES
            }
        )
        self.unique_decoders = nn.ModuleDict(
            {
                name: build_decoder(unique_dim, unique_hidden_dim, n_genes, dropout)
                for name in MODALITY_NAMES
            }
        )
        self.representation_names = REPRESENTATION_NAMES
        self.n_representations = len(self.representation_names)
        fusion_dim = shared_dim + len(MODALITY_NAMES) * unique_dim
        self.enable_residual_balanced_fusion = self.config["ENABLE_RESIDUAL_BALANCED_FUSION"]
        self.register_buffer("fusion_progress", torch.tensor(0.0, dtype=torch.float32))
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.LayerNorm(classifier_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 2, 1),
        )

    def _concat_representations(self, rep_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([rep_tensors[name] for name in self.representation_names], dim=1)

    def _branch_logits(self, rep_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        branch_logits = {}
        for branch_name in self.representation_names:
            masked_representations = {
                name: tensor if name == branch_name else torch.zeros_like(tensor)
                for (name, tensor) in rep_tensors.items()
            }
            branch_logits[branch_name] = self.classifier(
                self._concat_representations(masked_representations)
            )
        return branch_logits

    def _apply_icg_dominant_dropout(
        self,
        rep_tensors: Dict[str, torch.Tensor],
        branch_logits: Dict[str, torch.Tensor],
        targets: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if targets is None or not self.training or self.config["ICG_DROPOUT_BASE"] <= 0.0:
            return (rep_tensors, {})
        target_values = targets.to(device=next(iter(rep_tensors.values())).device).float()
        branch_confidences = []
        for name in self.representation_names:
            branch_prob = torch.sigmoid(branch_logits[name].squeeze(-1)).detach()
            correct_confidence = target_values * branch_prob + (1.0 - target_values) * (
                1.0 - branch_prob
            )
            branch_confidences.append(correct_confidence)
        confidence_matrix = torch.stack(branch_confidences, dim=1)
        other_mean = (confidence_matrix.sum(dim=1, keepdim=True) - confidence_matrix) / max(
            self.n_representations - 1, 1
        )
        rho = confidence_matrix / other_mean.clamp_min(1e-06)
        progress = self.fusion_progress.to(
            device=confidence_matrix.device, dtype=confidence_matrix.dtype
        ).clamp(0.0, 1.0)
        warmup = min(max(float(self.config["ICG_DROPOUT_WARMUP_RATIO"]), 0.0), 0.95)
        ramp = ((progress - warmup) / max(1.0 - warmup, 1e-06)).clamp(0.0, 1.0)
        drop_prob = self.config["ICG_DROPOUT_BASE"] * torch.tanh(F.relu(rho - 1.0)) * ramp
        drop_prob = drop_prob.clamp(max=self.config["ICG_DROPOUT_MAX"])
        if "shared" in self.representation_names:
            shared_idx = self.representation_names.index("shared")
            drop_prob[:, shared_idx] = (
                drop_prob[:, shared_idx] * self.config["ICG_SHARED_DROPOUT_SCALE"]
            )
        modulated_representations = {}
        for idx, name in enumerate(self.representation_names):
            keep_prob = (1.0 - drop_prob[:, idx : idx + 1]).clamp_min(1e-06)
            mask = torch.bernoulli(keep_prob).to(dtype=rep_tensors[name].dtype)
            modulated_representations[name] = rep_tensors[name] * mask / keep_prob
        return (
            modulated_representations,
            {
                "drop_prob": drop_prob,
                "rho": rho.detach(),
                "confidence": confidence_matrix.detach(),
                "dropout_ramp": ramp.detach(),
            },
        )

    def set_fusion_progress(self, training_progress: Optional[float]) -> None:
        if training_progress is None:
            return
        progress = float(min(max(training_progress, 0.0), 1.0))
        self.fusion_progress.fill_(progress)

    def _apply_residual_balanced_fusion(
        self, z_shared: torch.Tensor, z_unique: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        rep_tensors = {"shared": z_shared, **z_unique}
        raw_fusion = self._concat_representations(rep_tensors)
        if not self.enable_residual_balanced_fusion:
            return (raw_fusion, {"enabled": False, "final_representations": rep_tensors})
        device = raw_fusion.device
        dtype = raw_fusion.dtype
        rep_energy = torch.stack(
            [rep_tensors[name].pow(2).mean(dim=1).sqrt() for name in self.representation_names],
            dim=1,
        )
        first_layer_weight = self.classifier[0].weight
        block_strengths = []
        offset = 0
        for name in self.representation_names:
            width = rep_tensors[name].shape[1]
            block = first_layer_weight[:, offset : offset + width]
            block_strengths.append(block.pow(2).mean().sqrt())
            offset += width
        block_strength = torch.stack(block_strengths).to(device=device, dtype=dtype).view(1, -1)
        raw_contribution_proxy = rep_energy.detach() * block_strength.detach()
        raw_contribution = raw_contribution_proxy / raw_contribution_proxy.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-06)
        target_contribution = torch.full_like(raw_contribution, 1.0 / self.n_representations)
        contribution_gap = target_contribution - raw_contribution
        scale = (
            1.0
            + self.config["FUSION_WEAK_BOOST_BETA"] * torch.tanh(F.relu(contribution_gap))
            - self.config["FUSION_STRONG_SUPPRESSION_GAMMA"] * torch.tanh(F.relu(-contribution_gap))
        )
        scale = scale / scale.mean(dim=1, keepdim=True).clamp_min(1e-06)
        scale = scale.clamp(
            min=self.config["FUSION_SCALE_MIN"], max=self.config["FUSION_SCALE_MAX"]
        )
        balanced_representations = {
            name: rep_tensors[name] * scale[:, idx : idx + 1]
            for (idx, name) in enumerate(self.representation_names)
        }
        balanced_fusion = self._concat_representations(balanced_representations)
        progress = self.fusion_progress.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        warmup = min(max(float(self.config["FUSION_WARMUP_RATIO"]), 0.0), 0.95)
        ramp = ((progress - warmup) / max(1.0 - warmup, 1e-06)).clamp(0.0, 1.0)
        blend_weight = self.config["FUSION_FIXED_BLEND_WEIGHT"] * ramp
        fusion = raw_fusion + blend_weight * (balanced_fusion - raw_fusion)
        balanced_rep_energy = torch.stack(
            [
                balanced_representations[name].pow(2).mean(dim=1).sqrt()
                for name in self.representation_names
            ],
            dim=1,
        )
        balanced_contribution_proxy = balanced_rep_energy.detach() * block_strength.detach()
        balanced_contribution = balanced_contribution_proxy / balanced_contribution_proxy.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-06)
        final_rep_tensors = {
            name: rep_tensors[name]
            + blend_weight * (balanced_representations[name] - rep_tensors[name])
            for name in self.representation_names
        }
        final_rep_energy = torch.stack(
            [
                final_rep_tensors[name].pow(2).mean(dim=1).sqrt()
                for name in self.representation_names
            ],
            dim=1,
        )
        final_contribution_proxy = final_rep_energy.detach() * block_strength
        final_contribution = final_contribution_proxy / final_contribution_proxy.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-06)
        contribution = final_contribution
        concentration = contribution.pow(2).sum(dim=1)
        raw_concentration = raw_contribution.pow(2).sum(dim=1)
        balanced_concentration = balanced_contribution.pow(2).sum(dim=1)
        weak_participation_loss = (
            F.relu(self.config["FUSION_WEAK_PARTICIPATION_FLOOR"] - contribution[:, 1:])
            .pow(2)
            .mean()
        )
        return (
            fusion,
            {
                "enabled": True,
                "raw_contribution": raw_contribution,
                "contribution": contribution,
                "balanced_contribution": balanced_contribution,
                "scale": scale,
                "balanced_fusion": balanced_fusion,
                "blend_weight": blend_weight.detach(),
                "balance_ramp": ramp.detach(),
                "balance_loss_scale": ramp.detach(),
                "raw_concentration": raw_concentration,
                "concentration": concentration,
                "balanced_concentration": balanced_concentration,
                "weak_participation_loss": weak_participation_loss,
                "balanced_representations": balanced_representations,
                "final_representations": final_rep_tensors,
            },
        )

    def _read_pathway_semantic_memory(
        self, pathway_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pathway_activity = torch.stack(
            [pathway_inputs[name] for name in MODALITY_NAMES], dim=1
        ).mean(dim=1)
        semantic_attention = torch.softmax(
            pathway_activity.abs() / self.semantic_temperature, dim=1
        )
        semantic_context = semantic_attention @ self.pathway_semantic_memory
        z_shared_extra = self.semantic_encoder(semantic_context)
        return (z_shared_extra, semantic_attention, semantic_context)

    def forward(
        self,
        x_mrna: torch.Tensor,
        x_mirna: torch.Tensor,
        x_dnam: torch.Tensor,
        x_cnv: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        inputs = {"mrna": x_mrna, "mirna": x_mirna, "dnam": x_dnam, "cnv": x_cnv}
        pathway_inputs = {name: tensor @ self.gene_to_path for (name, tensor) in inputs.items()}
        shared_candidates = {
            name: self.shared_encoder(pathway) for (name, pathway) in pathway_inputs.items()
        }
        z_shared_intra = torch.stack(
            [shared_candidates[name] for name in MODALITY_NAMES], dim=1
        ).mean(dim=1)
        z_shared_extra = None
        semantic_attention = None
        semantic_context = None
        if self.use_pathway_semantic_memory:
            (z_shared_extra, semantic_attention, semantic_context) = (
                self._read_pathway_semantic_memory(pathway_inputs)
            )
            z_shared_final = self.semantic_fusion_norm(
                z_shared_intra + self.semantic_residual_scale * z_shared_extra
            )
        else:
            z_shared_final = z_shared_intra
        shared_pathway = {}
        shared_gene = {}
        z_unique = {}
        unique_gene = {}
        recon_gene = {}
        for name, tensor in inputs.items():
            shared_pathway[name] = self.shared_pathway_decoders[name](z_shared_intra)
            shared_gene[name] = shared_pathway[name] @ self.path_to_gene
            residual_gene = tensor - shared_gene[name]
            z_unique[name] = self.unique_encoders[name](residual_gene)
            unique_gene[name] = self.unique_decoders[name](z_unique[name])
            recon_gene[name] = shared_gene[name] + unique_gene[name]
        rep_tensors = {"shared": z_shared_final, **z_unique}
        branch_logits = self._branch_logits(rep_tensors)
        (modulated_rep_tensors, icg_dropout) = self._apply_icg_dominant_dropout(
            rep_tensors, branch_logits, targets
        )
        z_shared_for_fusion = modulated_rep_tensors["shared"]
        z_unique_for_fusion = {name: modulated_rep_tensors[name] for name in MODALITY_NAMES}
        (fusion, fusion_balance) = self._apply_residual_balanced_fusion(
            z_shared_for_fusion, z_unique_for_fusion
        )
        post_icg_representations = fusion_balance.get("final_representations")
        if not isinstance(post_icg_representations, dict):
            post_icg_representations = modulated_rep_tensors
        raw_logits = self.classifier(fusion)
        balanced_logits = None
        logits = raw_logits
        balanced_fusion = fusion_balance.get("balanced_fusion")
        if (
            self.enable_residual_balanced_fusion
            and isinstance(balanced_fusion, torch.Tensor)
            and (
                self.config["FUSION_BALANCED_AUX_LOSS_WEIGHT"] > 0.0
                or self.config["FUSION_BALANCED_CONSISTENCY_WEIGHT"] > 0.0
            )
        ):
            balanced_logits = self.classifier(balanced_fusion)
        return {
            "inputs": inputs,
            "pathway_inputs": pathway_inputs,
            "shared_candidates": shared_candidates,
            "shared_pathway": shared_pathway,
            "shared_gene": shared_gene,
            "z_shared": z_shared_intra,
            "z_shared_final": z_shared_final,
            "z_shared_extra": z_shared_extra,
            "pathway_semantic_attention": (
                semantic_attention.detach() if semantic_attention is not None else None
            ),
            "pathway_semantic_context": (
                semantic_context.detach() if semantic_context is not None else None
            ),
            "pathway_semantic_residual_scale": (
                self.semantic_residual_scale.detach()
                if self.semantic_residual_scale is not None
                else None
            ),
            "z_unique": z_unique,
            "post_icg_representations": post_icg_representations,
            "unique_gene": unique_gene,
            "recon_gene": recon_gene,
            "fusion": fusion,
            "fusion_balance": fusion_balance,
            "branch_logits": branch_logits,
            "icg_dropout": icg_dropout,
            "raw_logits": raw_logits,
            "balanced_logits": balanced_logits,
            "logits": logits,
        }
