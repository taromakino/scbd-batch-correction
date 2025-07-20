import lightning as L
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torchvision.models import resnet18, resnet50, densenet121
from scbd_batch_correction.utils.enum import EncoderType
from scbd_batch_correction.utils.hparams import HParams
from typing import Tuple


class SupConNet(nn.Module):
    def __init__(
            self, 
            img_channels: int, 
            encoder_type: EncoderType, 
            z_size: int,
    ) -> None:
        super().__init__()
        if encoder_type == EncoderType.RESNET18:
            self.encoder = resnet18()
            self.r_size = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
            if img_channels > 3:
                self.encoder.conv1 = nn.Conv2d(img_channels, 64, 7, stride=2, padding=3, bias=False)
        elif encoder_type == EncoderType.RESNET50:
            self.encoder = resnet50()
            self.r_size = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
            if img_channels > 3:
                self.encoder.conv1 = nn.Conv2d(img_channels, 64, 7, stride=2, padding=3, bias=False)
        else:
            assert encoder_type == EncoderType.DENSENET121
            self.encoder = densenet121()
            self.r_size = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Identity()
            if img_channels > 3:
                self.encoder.features[0] = nn.Conv2d(img_channels, 64, 7, stride=2, padding=3, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(self.r_size, self.r_size),
            nn.GELU(),
            nn.Linear(self.r_size, z_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        r = self.encoder(x)
        z = F.normalize(self.proj(r), dim=1)
        return z


class Model(L.LightningModule):
    def __init__(self, hparams: HParams) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.net_c = SupConNet(self.hparams.img_channels, self.hparams.encoder_type, self.hparams.z_size)
        self.net_s = SupConNet(self.hparams.img_channels, self.hparams.encoder_type, self.hparams.z_size)

    def get_inputs(self, batch: Tuple[Tensor, pd.DataFrame]) -> Tuple[Tensor, Tensor, Tensor]:
        x, df = batch
        y = torch.tensor(df.y.values, device=self.device)
        e = torch.tensor(df.e.values, device=self.device)
        return x, y, e

    def get_supcon_loss(self, z: Tensor, u: Tensor) -> Tensor:
        batch_size = len(z)
        u_col = u.unsqueeze(1)
        u_row = u.unsqueeze(0)
        mask = (u_col == u_row).float()
        offdiag_mask = 1. - torch.eye(batch_size, device=self.device)
        mask = mask * offdiag_mask
        logits = torch.matmul(z, z.T) / self.hparams.temperature
        p = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.)
        q = F.log_softmax(logits, dim=1)
        cross_entropy = F.cross_entropy(q, p)
        return cross_entropy

    def get_invariance_loss(self, zc: Tensor, e: Tensor) -> Tensor:
        batch_size = len(zc)
        e_col = e.unsqueeze(1)
        e_row = e.unsqueeze(0)
        mask_pos = (e_col == e_row).float()
        mask_neg = 1. - mask_pos
        offdiag_mask = 1. - torch.eye(batch_size, device=self.device)
        mask_pos = mask_pos * offdiag_mask
        logits = torch.matmul(zc, zc.T) / self.hparams.temperature
        q = F.log_softmax(logits, dim=1)
        log_prob_pos = (q * mask_pos).mean(dim=1)
        log_prob_neg = (q * mask_neg).mean(dim=1)
        return (log_prob_pos - log_prob_neg).abs().mean()

    def forward(self, x: Tensor, y: Tensor, e: Tensor) -> Tensor:
        zc = self.net_c(x)
        zs = self.net_s(x)
        supcon_loss_c = self.get_supcon_loss(zc, y)
        supcon_loss_s = self.get_supcon_loss(zs, e)
        invariance_loss = self.get_invariance_loss(zc, e)
        loss = supcon_loss_c + supcon_loss_s + self.hparams.alpha * invariance_loss
        return loss

    def training_step(self, batch: Tuple[Tensor, pd.DataFrame], batch_idx: int) -> Tensor:
        x, y, e = self.get_inputs(batch)
        loss = self(x, y, e)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Tuple[Tensor, pd.DataFrame], batch_idx: int) -> None:
        x, y, e = self.get_inputs(batch)
        loss = self(x, y, e)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def on_test_start(self) -> None:
        self.embed_path = os.path.join(self.hparams.results_dir, f"version_{self.hparams.seed}", "embed.csv")
        if os.path.exists(self.embed_path):
            os.remove(self.embed_path)
        os.makedirs(os.path.dirname(self.embed_path), exist_ok=True)

    def test_step(self, batch: Tuple[Tensor, pd.DataFrame], batch_idx: int) -> None:
        x, y, e = self.get_inputs(batch)
        zc = self.net_c(x)
        zc = zc.cpu().numpy()
        with open(self.embed_path, "a") as f:
            np.savetxt(f, zc, delimiter=",")

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)