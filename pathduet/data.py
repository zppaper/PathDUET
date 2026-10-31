from pathlib import Path
from functools import lru_cache
import csv
import hashlib

import numpy as np
import pandas as pd

from .constants import MODALITY_NAMES

MATRIX_FILES = ("mRNA.csv", "miRNA.csv", "dnam_gene.csv", "cnv_gene.csv")


def read_table(path, identifier):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or header[0] != identifier or len(header) != len(set(header)):
        raise ValueError(f"Invalid or duplicate CSV columns: {path}")
    frame = pd.read_csv(path, dtype={identifier: str}, keep_default_na=False,
                        float_precision="round_trip")
    if frame.empty or frame[identifier].eq("").any() or frame[identifier].duplicated().any():
        raise ValueError(f"Missing or duplicate identifiers: {path}")
    return frame


def digest_files(base, paths):
    result = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(base).as_posix()):
        result.update(path.relative_to(base).as_posix().encode("utf8") + b"\0")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(2**20), b""):
                digest.update(block)
        result.update(digest.digest())
    return result.hexdigest()


def resource_digest(path):
    path = Path(path)
    return digest_files(path, [path / "gene_pathway.csv", path / "semantic_memory.csv"])


def load_resources(path):
    path = Path(path)
    mapping = read_table(path / "gene_pathway.csv", "gene").set_index("gene")
    semantic = read_table(path / "semantic_memory.csv", "pathway").set_index("pathway")
    if mapping.columns.tolist() != semantic.index.tolist():
        raise ValueError("Semantic memory and pathway mapping orders differ.")
    mask = mapping.to_numpy(dtype=np.float32)
    memory = semantic.to_numpy(dtype=np.float32)
    if not np.isfinite(mask).all() or not np.isfinite(memory).all() or (mask < 0).any():
        raise ValueError("Invalid pathway or semantic resource values.")
    if mask.shape[1] == 0 or memory.shape[1] == 0:
        raise ValueError("Empty pathway or semantic resources.")
    return pd.DataFrame(mask, index=mapping.index, columns=mapping.columns), memory


def cohort_signature(directory, with_labels):
    files = list(MATRIX_FILES) + (["label.csv"] if with_labels else [])
    return tuple((name, (directory / name).stat().st_size,
                  (directory / name).stat().st_mtime_ns) for name in files)


@lru_cache(maxsize=2)
def read_cohort_cached(directory, with_labels, signature):
    directory = Path(directory)
    arrays = []
    ids = None
    genes = None
    for name in MATRIX_FILES:
        frame = read_table(directory / name, "sample_id").set_index("sample_id")
        if ids is None:
            ids = frame.index.to_numpy(dtype=str)
            genes = frame.columns.to_numpy(dtype=str)
        elif not np.array_equal(ids, frame.index.to_numpy(dtype=str)) or not np.array_equal(
                genes, frame.columns.to_numpy(dtype=str)):
            raise ValueError("Omics sample or gene orders differ.")
        values = frame.to_numpy(dtype=np.float32)
        if values.shape != (len(ids), len(genes)) or not np.isfinite(values).all():
            raise ValueError(f"Invalid molecular matrix: {name}")
        values.flags.writeable = False
        arrays.append(values)
    labels = None
    if with_labels:
        frame = read_table(directory / "label.csv", "sample_id")
        if frame.columns.tolist() != ["sample_id", "label"] or not np.array_equal(
                frame.sample_id.to_numpy(dtype=str), ids):
            raise ValueError("Label order differs from molecular sample order.")
        labels = frame.label.to_numpy()
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Labels must be binary integers.")
        labels = labels.astype(np.int64)
    return arrays, labels, ids, genes


def load_cohort(directory, genes, with_labels=True):
    directory = Path(directory).resolve()
    arrays, labels, ids, actual_genes = read_cohort_cached(
        str(directory), with_labels, cohort_signature(directory, with_labels))
    if not np.array_equal(actual_genes, np.asarray(genes, dtype=str)):
        raise ValueError("Dataset gene order differs from the resources.")
    return arrays, labels, ids


def read_scaler(path, genes):
    frame = read_table(path, "gene")
    expected = ["gene"] + [field for name in MODALITY_NAMES
                           for field in (f"mean_{name}", f"scale_{name}")]
    if frame.columns.tolist() != expected or frame.gene.tolist() != list(genes):
        raise ValueError("Scaler columns or gene order differ from the resources.")
    parameters = []
    for name in MODALITY_NAMES:
        mean = frame[f"mean_{name}"].to_numpy(dtype=np.float64)
        scale = frame[f"scale_{name}"].to_numpy(dtype=np.float64)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Invalid saved training-fold scaler.")
        parameters.append((mean, scale))
    return parameters


def standardize(arrays, parameters, indices=None):
    result = []
    for values, (mean, scale) in zip(arrays, parameters):
        transformed = values.copy() if indices is None else values[indices].copy()
        transformed -= mean
        transformed /= scale
        if not np.isfinite(transformed).all():
            raise ValueError("Standardized input contains nonfinite values.")
        result.append(transformed)
    return result


def fold_files(path):
    path = Path(path)
    cohort = path.parents[2]
    return cohort, [cohort / name for name in MATRIX_FILES] + [cohort / "label.csv",
                    path, path.with_name(path.stem + "_scaler.csv")]


def fold_digest(path):
    cohort, files = fold_files(path)
    return digest_files(cohort, files)


def load_fold(path, genes):
    path = Path(path)
    cohort, _ = fold_files(path)
    arrays, labels, ids = load_cohort(cohort, genes)
    split = read_table(path, "sample_id")
    if split.columns.tolist() != ["sample_id", "cohort_index", "partition", "partition_order"]:
        raise ValueError("Invalid split columns.")
    if set(split.partition) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test partitions.")
    for column in ("cohort_index", "partition_order"):
        if not pd.api.types.is_integer_dtype(split[column]) or (split[column] < 0).any():
            raise ValueError(f"Invalid split indices: {column}")
    indices = split.cohort_index.to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(indices), np.arange(len(ids))):
        raise ValueError("Fold must cover every cohort row exactly once.")
    if not np.array_equal(split.sample_id.to_numpy(dtype=str), ids[indices]):
        raise ValueError("Split sample IDs differ from their cohort row indices.")
    parameters = read_scaler(path.with_name(path.stem + "_scaler.csv"), genes)
    result = {}
    for name in ("train", "validation", "test"):
        part = split[split.partition == name].sort_values("partition_order")
        if not np.array_equal(part.partition_order.to_numpy(), np.arange(len(part))):
            raise ValueError("Partition order must be unique and contiguous.")
        ix = part.cohort_index.to_numpy(dtype=np.int64)
        if set(np.unique(labels[ix])) != {0, 1}:
            raise ValueError(f"Both classes are required in {name}.")
        result[name] = {"arrays": standardize(arrays, parameters, ix),
                        "labels": labels[ix].copy(), "ids": ids[ix].copy(),
                        "indices": ix}
    return result


def load_inputs(directory, genes, scaler):
    arrays, _, ids = load_cohort(directory, genes, with_labels=False)
    return standardize(arrays, read_scaler(scaler, genes)), ids.copy()
