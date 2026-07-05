from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def read_matrix(path: str | Path, transpose: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", index_col=0)
    return df.T if transpose else df


def load_label_file(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_gmt(path: str | Path) -> Dict[str, List[str]]:
    pathways = {}
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                pathways[parts[0]] = [gene for gene in parts[2:] if gene]
    return pathways


def build_gene_pathway_mask(anchor_genes: Sequence[str], pathway_genes: Dict[str, List[str]]) -> pd.DataFrame:
    anchor_index = pd.Index(anchor_genes)
    mask = pd.DataFrame(0, index=anchor_index, columns=list(pathway_genes.keys()), dtype="int8")
    for pathway, genes in pathway_genes.items():
        present = anchor_index.intersection(genes)
        mask.loc[present, pathway] = 1
    return mask


def align_samples(*dfs: pd.DataFrame, label_df: pd.DataFrame) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    sample_ids = set(label_df.iloc[:, 0].astype(str))
    for df in dfs:
        sample_ids &= set(df.index.astype(str))
    ordered = sorted(sample_ids)
    aligned_dfs = [df.loc[ordered] for df in dfs]
    aligned_labels = label_df[label_df.iloc[:, 0].astype(str).isin(ordered)].copy()
    aligned_labels = aligned_labels.set_index(aligned_labels.columns[0]).loc[ordered].reset_index()
    return aligned_dfs, aligned_labels


def standardize_modalities(
    train_arrays: Sequence[np.ndarray],
    val_arrays: Sequence[np.ndarray],
    test_arrays: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    train_scaled, val_scaled, test_scaled = [], [], []
    for train_array, val_array, test_array in zip(train_arrays, val_arrays, test_arrays):
        scaler = StandardScaler()
        train_scaled.append(scaler.fit_transform(train_array).astype(np.float32))
        val_scaled.append(scaler.transform(val_array).astype(np.float32))
        test_scaled.append(scaler.transform(test_array).astype(np.float32))
    return train_scaled, val_scaled, test_scaled


def make_loader(arrays: Sequence[np.ndarray], y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = [torch.tensor(array, dtype=torch.float32) for array in arrays]
    labels = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(*tensors, labels), batch_size=batch_size, shuffle=shuffle)
