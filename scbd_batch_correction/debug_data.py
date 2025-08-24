import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

from scbd_batch_correction.data.cellpainting2 import IMG_CHANNELS, IMG_ORIGINAL_PIXELS, LMDB_DIRNAME, get_df
from scbd_batch_correction.utils.data import Arcsinh


def main(args):
    df = get_df(args.data_dir)

    results_df = {
        "img_name": [],
        "pre_uint16_min": [],
        "pre_uint16_max": [],
        "pre_float16_min": [],
        "pre_float16_max": [],
        "post_uint16_min": [],
        "post_uint16_max": [],
        "post_float16_min": [],
        "post_float16_max": [],
    }

    lmdb = lmdb.Environment(os.path.join(args.data_dir, LMDB_DIRNAME), readonly=True, readahead=False, lock=False, subdir=False)

    transforms = T.Compose([
        T.Resize((args.img_pixels, args.img_pixels)),
        Arcsinh(),
        T.Normalize(7., 7.),
    ])

    for idx in range(len(df)):
        row = df.iloc[idx]
        img_name = f'{row["Well"]}_{row["Gene"]}_{row["index"]}'
        results_df["image_name"].append(img_name)
        with lmdb.begin(write=False, buffers=True) as txn:
            buf = txn.get(img_name.encode())
            x_uint16 = np.frombuffer(buf, dtype="uint16")
            x_float16 = np.frombuffer(buf, dtype=np.float16)
        x_uint16 = x_uint16.reshape((IMG_CHANNELS, IMG_ORIGINAL_PIXELS, IMG_ORIGINAL_PIXELS))
        x_float16 = x_float16.reshape((IMG_CHANNELS, IMG_ORIGINAL_PIXELS, IMG_ORIGINAL_PIXELS))
        x_uint16 = torch.tensor(x_uint16)
        x_float16 = torch.tensor(x_float16)
        results_df["pre_uint16_min"].append(x_uint16.min().item())
        results_df["pre_uint16_max"].append(x_uint16.max().item())
        results_df["pre_float16_min"].append(x_float16.min().item())
        results_df["pre_float16_max"].append(x_float16.max().item())
        x_uint16 = transforms(x_uint16)
        x_float16 = transforms(x_float16)
        results_df["post_uint16_min"].append(x_uint16.min().item())
        results_df["post_uint16_max"].append(x_uint16.max().item())
        results_df["post_float16_min"].append(x_float16.min().item())
        results_df["post_float16_max"].append(x_float16.max().item())
    
    results_df = pd.DataFrame(results_df)
    results_df.to_parquet(os.path.join(args.results_dir, "debug_data.parquet"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--img_pixels", type=int, default=64)