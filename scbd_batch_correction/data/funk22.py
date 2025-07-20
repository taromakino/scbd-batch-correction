import lmdb
import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from scbd_batch_correction.utils.data import Arcsinh, get_y_batch_dataloader, get_dataloader
from scbd_batch_correction.utils.enum import ExperimentGroup
from scbd_batch_correction.utils.hparams import HParams
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


IMG_CHANNEL_NAMES = [
    "DNA damage",
    "F-actin",
    "DNA content",
    "Microtubules",
]
PLATES = [
    "20200202_6W-LaC024A",
    "20200202_6W-LaC024D",
    "20200202_6W-LaC024E",
    "20200202_6W-LaC024F",
    "20200206_6W-LaC025A",
    "20200206_6W-LaC025B",
]
PH_DIMS = (2960, 2960)
VAL_RATIO = 0.02
IMG_ORIGINAL_PIXELS = 100
IMG_CHANNELS = len(IMG_CHANNEL_NAMES)


class Funk22Dataset(Dataset):
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
            T.Normalize(7., 7.),
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


def get_lmdb_treatment_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "funk22_lmdb_shuffled", "perturbed")


def get_lmdb_control_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "funk22_lmdb_shuffled", "ntc")


def get_y(df: pd.DataFrame) -> pd.Series:
    """
    Gets the integer values of the perturbed genes
    """
    y_names_unique = sorted(df.gene_symbol_0.unique())
    y_name_to_idx = {y_name: i for i, y_name in enumerate(y_names_unique)}
    y = df.gene_symbol_0.map(y_name_to_idx)
    return y


def get_e(df: pd.DataFrame) -> pd.Series:
    """
    Gets the integer values of the batch. Here, the batch is defined as the plate and well pair. In Funk22, there are
    six plates with six wells each. Since two wells are unused in one of the plates, there are 34 plate and well pairs.
    """
    e_names = df["plate"] + df["well"]
    e_names_unique = sorted(e_names.unique())
    e_name_to_idx = {batch_name: i for i, batch_name in enumerate(e_names_unique)}
    e = e_names.map(e_name_to_idx)
    return e


def get_group_df(dir: str, group: ExperimentGroup) -> pd.DataFrame:
    """
    Get the metadata from the lmdb directory. There is some data filtering done that was originally done in
    https://www.biorxiv.org/content/10.1101/2023.11.28.569094v2.abstract.
    """
    df = pd.read_csv(os.path.join(dir, "key.csv"), dtype={"UID": str})
    df = df[df.plate.isin(PLATES)]
    radius = IMG_ORIGINAL_PIXELS / 2
    df = df[
        df.cell_i.between(radius, PH_DIMS[0] - radius) &
        df.cell_j.between(radius, PH_DIMS[1] - radius)
    ]
    df["group"] = group
    return df


def get_df(data_dir: str) -> pd.DataFrame:
    """
    Get the metadata for the treatment and control groups, and combine them into a single dataframe. Add the "y" column
    which are the integer values of the perturbed gene, and the "e" column which are the integer values of the batch.
    """
    df_treatment = get_group_df(get_lmdb_treatment_dir(data_dir), ExperimentGroup.TREATMENT)
    df_control = get_group_df(get_lmdb_control_dir(data_dir), ExperimentGroup.CONTROL)
    df = pd.concat((df_treatment, df_control))
    df.reset_index(drop=True, inplace=True)
    df["y"] = get_y(df)
    df["e"] = get_e(df)
    df = df.sample(frac=1, random_state=0)
    return df


def get_data(hparams: HParams) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Get the train, val, and all dataloaders, along with metadata. The train and val dataloaders are used for model
    training, while the all dataloader is used to compute embeddings for the entire dataset.
    """
    df = get_df(hparams.data_dir)
    df_train, df_val = train_test_split(df, test_size=VAL_RATIO)

    lmdb_treatment_dir = get_lmdb_treatment_dir(hparams.data_dir)
    lmdb_control_dir = get_lmdb_control_dir(hparams.data_dir)

    dataset_train = Funk22Dataset(
        df_train,
        lmdb_treatment_dir,
        lmdb_control_dir,
        IMG_CHANNELS,
        IMG_ORIGINAL_PIXELS,
        hparams.img_pixels,
    )

    dataset_val = Funk22Dataset(
        df_val,
        lmdb_treatment_dir,
        lmdb_control_dir,
        IMG_CHANNELS,
        IMG_ORIGINAL_PIXELS,
        hparams.img_pixels,
    )

    dataset_all = Funk22Dataset(
        df,
        lmdb_treatment_dir,
        lmdb_control_dir,
        IMG_CHANNELS,
        IMG_ORIGINAL_PIXELS,
        hparams.img_pixels,
    )

    y_train = df_train.y.values
    pmf_y = np.bincount(y_train)
    pmf_y = pmf_y / pmf_y.sum()

    data_train = get_y_batch_dataloader(dataset_train, pmf_y, hparams.batch_size, hparams.y_per_batch, hparams.workers)
    data_val = get_y_batch_dataloader(dataset_val, pmf_y, hparams.batch_size, hparams.y_per_batch, hparams.workers)
    data_all = get_dataloader(dataset_all,hparams.batch_size, hparams.workers)

    metadata = {
        "img_channels": IMG_CHANNELS,
        "y_size": df.y.max() + 1,
        "e_size": df.e.max() + 1
    }

    return data_train, data_val, data_all, metadata