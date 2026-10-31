from typing import Dict, Sequence, Tuple
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


