
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import Dataset
from scbd_batch_correction.utils.const import UINT8_MAX
from scbd_batch_correction.utils.data import get_dataloader


RNG = np.random.RandomState(0)
TRAIN_RATIO = 0.8

IMG_PIXELS = 32
IMG_CHANNELS = 3
Y_SIZE = 10
E_SIZE = 2


class CMNISTDataset(Dataset):
    def __init__(self, x, df):
        super().__init__()
        self.x = x
        self.df = df

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.df.iloc[idx]


def set_brightness(x, y, e, y_value, e_value):
    mask = (y == y_value) & (e == e_value)
    mask_size = mask.sum().item()
    if e_value == 0:
        other_mult = y_value / Y_SIZE
        x_mult = torch.tensor([1., other_mult, other_mult])
    else:
        other_mult = (Y_SIZE - 1 - y_value) / Y_SIZE
        x_mult = torch.tensor([other_mult, 1., other_mult])
    x_mult = x_mult.unsqueeze(0).repeat_interleave(mask_size, dim=0)
    x_mult = x_mult.view(mask_size, IMG_CHANNELS, 1, 1)
    x[mask] = x[mask] * x_mult


def get_tensors(data_dir, is_trainval):
    mnist = datasets.MNIST(data_dir, train=is_trainval, download=True)
    x, y = mnist.data, mnist.targets
    dataset_size = len(x)

    x = x.unsqueeze(1)
    x = x.repeat_interleave(IMG_CHANNELS, 1)
    x = F.interpolate(x, size=(32, 32))
    x = x / UINT8_MAX

    if is_trainval:
        # As y increases, color goes from red to light red in e=0, and from light green to green in e=1
        e = torch.randint(0, E_SIZE, (dataset_size,))
        for e_value in range(E_SIZE):
            for y_value in range(Y_SIZE):
                set_brightness(x, y, e, y_value, e_value)
    else:
        # Digits are white
        e = torch.full((dataset_size,), torch.nan, dtype=torch.float32)

    df = pd.DataFrame({"y": y.numpy(), "e": e.numpy()})
    return x, df


def get_data(hparams):
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(0)
    x_trainval, df_trainval = get_tensors(hparams.data_dir, True)
    x_test, df_test = get_tensors(hparams.data_dir, False)
    torch.random.set_rng_state(rng_state)
    trainval_size = len(x_trainval)
    train_size = int(TRAIN_RATIO * trainval_size)
    train_idxs = RNG.choice(np.arange(trainval_size), train_size, replace=False)
    val_idxs = np.setdiff1d(np.arange(trainval_size), train_idxs)

    x_train, df_train = x_trainval[train_idxs], df_trainval.iloc[train_idxs]
    x_val, df_val = x_trainval[val_idxs], df_trainval.iloc[val_idxs]

    dataset_train = CMNISTDataset(x_train, df_train)
    dataset_val = CMNISTDataset(x_val, df_val)
    dataset_test = CMNISTDataset(x_test, df_test)

    data_train = get_dataloader(dataset_train, hparams.batch_size, hparams.workers)
    data_val = get_dataloader(dataset_val, hparams.batch_size, hparams.workers)
    data_test = get_dataloader(dataset_test, hparams.batch_size, hparams.workers)

    metadata = {
        "img_channels": IMG_CHANNELS,
        "y_size": Y_SIZE,
        "e_size": E_SIZE
    }
    return data_train, data_val, data_test, metadata