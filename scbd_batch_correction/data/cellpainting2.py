import lmdb
import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from scbd_batch_correction.utils.data import Arcsinh, get_dataloader
from scbd_batch_correction.utils.hparams import HParams
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


IMG_CHANNEL_NAMES = [

]
PLATES = [

]
VAL_RATIO = 0.02
IMG_ORIGINAL_PIXELS = 64
IMG_CHANNELS = len(IMG_CHANNEL_NAMES)


class CellPainting2Dataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            data_dir: str,
            img_channels: int,
            img_original_pixels: int,
            img_pixels: int,
    ):
        self.df = df
        self.img_channels = img_channels
        self.img_original_pixels = img_original_pixels
        self.lmdb = lmdb.Environment(data_dir, readonly=True, readahead=False, lock=False)
        self.transforms = T.Compose([
            T.Resize((img_pixels, img_pixels)),
            Arcsinh(),
            T.Normalize(7., 7.),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, pd.Series]:
        row = self.df.iloc[idx]
        img_name = f'{row.UID}_{row.plate}_{row.well}_{row.tile}_{row.gene_symbol_0}_{row["index"]}'
        with self.lmdb.begin(write=False, buffers=True) as txn:
            buf = txn.get(img_name.encode())
            x = np.frombuffer(buf, dtype="uint16")
        x = x.reshape((self.img_channels, self.img_original_pixels, self.img_original_pixels))
        x = torch.tensor(x)
        x = self.transforms(x)
        return x, row


def get_y(df: pd.DataFrame) -> pd.Series:
    """
    Gets the integer values of the perturbed genes
    """
    raise NotImplementedError


def get_e(df: pd.DataFrame) -> pd.Series:
    """
    Gets the integer values of the batch. Here, the batch is defined as the plate and well pair. In Funk22, there are
    six plates with six wells each. Since two wells are unused in one of the plates, there are 34 plate and well pairs.
    """
    raise NotImplementedError


def get_df(data_dir: str) -> pd.DataFrame:
    """
    Get the metadata for the treatment and control groups, and combine them into a single dataframe. Add the "y" column
    which are the integer values of the perturbed gene, and the "e" column which are the integer values of the batch.
    """
    df = pd.read_pickle(os.path.join(data_dir))
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

    dataset_train = CellPainting2Dataset(
        df_train,
        hparams.data_dir,
        IMG_CHANNELS,
        IMG_ORIGINAL_PIXELS,
        hparams.img_pixels,
    )

    dataset_val = CellPainting2Dataset(
        df_val,
        hparams.data_dir,
        IMG_CHANNELS,
        IMG_ORIGINAL_PIXELS,
        hparams.img_pixels,
    )

    dataset_all = CellPainting2Dataset(
        df,
        hparams.data_dir,
        IMG_CHANNELS,
        IMG_ORIGINAL_PIXELS,
        hparams.img_pixels,
    )

    y_train = df_train.y.values
    pmf_y = np.bincount(y_train)
    pmf_y = pmf_y / pmf_y.sum()

    data_train = get_dataloader(dataset_train, True, pmf_y, hparams.batch_size, hparams.y_per_batch, hparams.workers)
    data_val = get_dataloader(dataset_val, True, pmf_y, hparams.batch_size, hparams.y_per_batch, hparams.workers)
    data_all = get_dataloader(dataset_all, False, pmf_y, hparams.batch_size, hparams.y_per_batch, hparams.workers)

    metadata = {
        "img_channels": IMG_CHANNELS,
        "y_size": df.y.max() + 1,
        "e_size": df.e.max() + 1
    }

    return data_train, data_val, data_all, metadata