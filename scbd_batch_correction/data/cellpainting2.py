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
    "1-Channel:0:0", "1-Channel:0:1", "1-Channel:0:2", "1-Channel:0:3", "1-Channel:0:4",
    "2-Channel:0:0", "2-Channel:0:1", "2-Channel:0:2", "2-Channel:0:3", "2-Channel:0:4",
    "3-Channel:0:0", "3-Channel:0:1", "3-Channel:0:2", "3-Channel:0:3", "3-Channel:0:4",
    "4-Channel:0:0", "4-Channel:0:1", "4-Channel:0:2", "4-Channel:0:3", "4-Channel:0:4"
]
WELLS = [ 
    "A1",
    "A2",
    "A3",
    "B1",
    "B2",
    "B3"
]
VAL_RATIO = 0.02
IMG_ORIGINAL_PIXELS = 64
IMG_CHANNELS = len(IMG_CHANNEL_NAMES)

# File names for data loading
METADATA_FILENAME = "lmdb.pkl"
LMDB_DIRNAME = "cell_crops.lmdb"


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
        self.lmdb = lmdb.Environment(os.path.join(data_dir, LMDB_DIRNAME), readonly=True, readahead=False, lock=False, subdir=False)
        self.transforms = T.Compose([
            T.Resize((img_pixels, img_pixels)),
            Arcsinh(),
            T.Normalize(7., 7.),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, pd.Series]:
        row = self.df.iloc[idx]
        img_name = f'{row["Well"]}_{row["Gene"]}_{row["index"]}'
        with self.lmdb.begin(write=False, buffers=True) as txn:
            buf = txn.get(img_name.encode())
            # ACL: edit bc input is float16 but weights are different type
            x = np.frombuffer(buf, dtype=np.float16)
            x = x.astype(np.float32)
            # ACL: end of edit
        x = x.reshape((self.img_channels, self.img_original_pixels, self.img_original_pixels))
        x = torch.tensor(x)
        x = self.transforms(x)
        return x, row


def get_y(df: pd.DataFrame) -> pd.Series:
    """
    Gets the integer values of the perturbed genes
    """
    # Get unique genes and create a mapping to integers
    y_names_unique = sorted(df['Gene'].unique())
    y_name_to_idx = {y_name: i for i, y_name in enumerate(y_names_unique)}
    y = df['Gene'].map(y_name_to_idx)
    return y


def get_e(df: pd.DataFrame) -> pd.Series:
    """
    Gets the integer values of the batch. Here, the batch is defined as WELL ONLY.
    """
    e_names = df["Well"]
    e_names_unique = sorted(e_names.unique())
    e_name_to_idx = {batch_name: i for i, batch_name in enumerate(e_names_unique)}
    e = e_names.map(e_name_to_idx)
    return e


def get_df(data_dir: str) -> pd.DataFrame:
    """
    Get the metadata for the treatment and control groups, and combine them into a single dataframe. Add the "y" column
    which are the integer values of the perturbed gene, and the "e" column which are the integer values of the batch.
    """
    metadata_path = os.path.join(data_dir, METADATA_FILENAME)
    df = pd.read_pickle(metadata_path)
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