import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, List
from utils.const import UINT32_MAX


class Arcsinh(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.arcsinh(x)


class YBatchSampler(Sampler):
    def __init__(
            self,
            y: Tensor,
            pmf_y: np.ndarray,
            batch_size: int,
            y_per_batch: int,
    ):
        super().__init__()
        assert batch_size % y_per_batch == 0
        self.y = y
        self.pmf_y = pmf_y
        self.y_per_batch = y_per_batch
        self.batch_size_per_y = batch_size // y_per_batch
        self.y_unique = np.unique(y)
        self.y_to_idxs = {y_value.item(): np.where(y == y_value)[0] for y_value in self.y_unique}

    def __iter__(self):
        while True:
            y_batch = np.random.choice(self.y_unique, size=self.y_per_batch, replace=False, p=self.pmf_y)
            idxs = []
            for y_value in y_batch:
                y_idxs = self.y_to_idxs[y_value]
                if self.batch_size_per_y < len(y_idxs):
                    idxs += np.random.choice(y_idxs, size=self.batch_size_per_y, replace=False).tolist()
                else:
                    idxs += y_idxs.tolist()
            yield idxs

    def __len__(self):
        return UINT32_MAX


def collate(batch: List[Tuple[torch.Tensor, pd.Series]]) -> Tuple[torch.Tensor, pd.DataFrame]:
    tensor_tuple, series_tuple = zip(*batch)
    x = torch.stack(tensor_tuple, dim=0)
    df = pd.concat(series_tuple, axis=1).T
    df.y = df.y.astype("int64")
    if df.e.isna().any():
        df.e = df.e.astype("float32")
    else:
        df.e = df.e.astype("int64")
    return x, df


def get_dataloader(
        dataset: Dataset,
        is_y_batch: bool,
        pmf_y: np.ndarray,
        batch_size: int,
        y_per_batch: int,
        workers: int,
) -> DataLoader:
    if is_y_batch:
        batch_sampler = YBatchSampler(dataset.df.y.values, pmf_y, batch_size, y_per_batch)
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=workers, collate_fn=collate, pin_memory=True,
            persistent_workers=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, num_workers=workers, collate_fn=collate, pin_memory=True,
            persistent_workers=True)