import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from scbd_batch_correction.utils.enum import ExperimentGroup
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import Tuple, List
from utils.const import UINT32_MAX


class Arcsinh(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.arcsinh(x)


class OpticalPooledScreeningDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            lmdb_treatment_dir: str,
            lmdb_control_dir: str,
            img_channels: int,
            img_original_pixels: int,
            img_pixels: int,
    ):
        self.df = df
        self.img_channels = img_channels
        self.img_original_pixels = img_original_pixels
        self.lmdb_treatment = lmdb.Environment(lmdb_treatment_dir, readonly=True, readahead=False, lock=False)
        self.lmdb_control = lmdb.Environment(lmdb_control_dir, readonly=True, readahead=False, lock=False)
        self.transforms = T.Compose([
            T.Resize((img_pixels, img_pixels)),
            Arcsinh(),
            T.Normalize(7., 7.)
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, pd.Series]:
        row = self.df.iloc[idx]
        is_treatment = self.df.group.iloc[idx] == ExperimentGroup.TREATMENT
        lmdb = self.lmdb_treatment if is_treatment else self.lmdb_control
        img_name = f'{row.UID}_{row.plate}_{row.well}_{row.tile}_{row.gene_symbol_0}_{row["index"]}'
        with lmdb.begin(write=False, buffers=True) as txn:
            buf = txn.get(img_name.encode())
            x = np.frombuffer(buf, dtype="uint16")
        x = x.reshape((self.img_channels, self.img_original_pixels, self.img_original_pixels))
        x = torch.tensor(x)
        x = self.transforms(x)
        return x, row


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
        dataset: OpticalPooledScreeningDataset,
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