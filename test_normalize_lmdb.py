import lmdb
import numpy as np
import torch
from typing import Tuple
import matplotlib.pyplot as plt
from typing import Optional, Union

def plot_distributions(raw_img: Union[np.ndarray, torch.Tensor], 
                      normalized_img: torch.Tensor,
                      channel: Optional[int] = None,
                      save_path: Optional[str] = None):
    
    plt.figure(figsize=(15, 5))
    
    # Plot raw distribution
    plt.subplot(121)
    if channel is not None:
        plt.hist(raw_img[channel].flatten(), bins=100, alpha=0.7)
        plt.title(f'Raw pixel distribution (channel {channel})')
    else:
        for i in range(raw_img.shape[0]):
            plt.hist(raw_img[i].flatten(), bins=100, alpha=0.3, label=f'Channel {i}')
        plt.legend()
        plt.title('Raw pixel distribution (all channels)')
    plt.xlabel('Pixel value')
    plt.ylabel('Count')
    
    # Plot normalized distribution
    plt.subplot(122)
    if channel is not None:
        plt.hist(normalized_img[channel].numpy().flatten(), bins=100, alpha=0.7)
        plt.title(f'Normalized pixel distribution (channel {channel})')
    else:
        for i in range(normalized_img.shape[0]):
            plt.hist(normalized_img[i].numpy().flatten(), bins=100, alpha=0.3, label=f'Channel {i}')
        plt.legend()
        plt.title('Normalized pixel distribution (all channels)')
    plt.xlabel('Pixel value')
    plt.ylabel('Count')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def normalize_image(x: np.ndarray, img_size: int = None) -> torch.Tensor:

    # Convert to torch tensor
    x = torch.tensor(x, dtype=torch.float16)
    
    # Resize if needed
    if img_size is not None and (x.shape[1] != img_size or x.shape[2] != img_size):
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), 
            size=(img_size, img_size), 
            mode='bilinear'
        ).squeeze(0)
    
    # Apply arcsinh transform
    x = torch.arcsinh(x)
    
    # Apply Gaussian normalization with mean=7, std=7 (as in original code)
    x = (x - 7.0) / 7.0
    
    return x

def process_lmdb(lmdb_path: str, img_channels: int, img_pixels: int) -> None:

    # Open LMDB environment
    env = lmdb.Environment(lmdb_path, readonly=True, readahead=False, subdir=False, lock=False)
    
    # Start read transaction
    with env.begin(write=False) as txn:
        # Get cursor to iterate through all items
        cursor = txn.cursor()
        
        for idx, (key, value) in enumerate(cursor):
            # Convert bytes to image array
            img = np.frombuffer(value, dtype=np.float16)
            img = img.reshape((img_channels, img_pixels, img_pixels))
            
            # Apply normalization
            normalized_img = normalize_image(img)
            
            # Print stats
            print(f"Key: {key.decode()}")
            print(f"Normalized image stats - Min: {normalized_img.min():.3f}, Max: {normalized_img.max():.3f}, Mean: {normalized_img.mean():.3f}")
            
            # Plot distributions
            # All channels
            plot_distributions(img, normalized_img, save_path=f'distributions_all_channels_{idx}.png')
            
            # DAPI1 just to see one channel isolated
            plot_distributions(img, normalized_img, channel=0, save_path=f'distributions_channel0_{idx}.png')
            
            # Only process first few images
            if idx >= 2:
                break

if __name__ == "__main__":
    
    LMDB_PATH = "/gstore/data/marioni_group/Carolina/CP2.0/CellCrops/cell_crops_500_test/cell_crops.lmdb"
    IMG_CHANNELS = 20
    IMG_PIXELS = 64
    
    process_lmdb(LMDB_PATH, IMG_CHANNELS, IMG_PIXELS)