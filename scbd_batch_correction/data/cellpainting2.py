import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.enum import ExperimentGroup
from scbd_batch_correction.utils.data import OpticalPooledScreeningDataset, get_dataloader
from scbd_batch_correction.utils.hparams import HParams
from torch.utils.data import DataLoader
from typing import Tuple


IMG_CHANNEL_NAMES = [

]
PLATES = [

]
VAL_RATIO = 0.02
IMG_ORIGINAL_PIXELS = None
IMG_CHANNELS = len(IMG_CHANNEL_NAMES)


def get_lmdb_treatment_dir(data_dir: str) -> str:
    raise NotImplementedError


def get_lmdb_control_dir(data_dir: str) -> str:
    raise NotImplementedError


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


def get_group_df(dir: str, group: ExperimentGroup) -> pd.DataFrame:
    """
    Get the metadata from the lmdb directory.
    """
    raise NotImplementedError


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

    dataset_train = OpticalPooledScreeningDataset(
        df_train,
        lmdb_treatment_dir,
        lmdb_control_dir,
        IMG_CHANNELS,
        IMG_ORIGINAL_PIXELS,
        hparams.img_pixels,
    )

    dataset_val = OpticalPooledScreeningDataset(
        df_val,
        lmdb_treatment_dir,
        lmdb_control_dir,
        IMG_CHANNELS,
        IMG_ORIGINAL_PIXELS,
        hparams.img_pixels,
    )

    dataset_all = OpticalPooledScreeningDataset(
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

    data_train = get_dataloader(dataset_train, True, pmf_y, hparams.batch_size, hparams.y_per_batch, hparams.workers)
    data_val = get_dataloader(dataset_val, True, pmf_y, hparams.batch_size, hparams.y_per_batch, hparams.workers)
    data_all = get_dataloader(dataset_all, False, pmf_y, hparams.batch_size, hparams.y_per_batch, hparams.workers)

    metadata = {
        "img_channels": IMG_CHANNELS,
        "y_size": df.y.max() + 1,
        "e_size": df.e.max() + 1
    }

    return data_train, data_val, data_all, metadata