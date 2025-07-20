import lightning as L
import scbd_batch_correction.data.cmnist as cmnist
import scbd_batch_correction.data.funk22 as funk22
import scbd_batch_correction.data.cellpainting2 as cellpainting2
from argparse import ArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from scbd_batch_correction.model import Model
from scbd_batch_correction.utils.enum import Dataset, EncoderType
from scbd_batch_correction.utils.hparams import HParams


def main(hparams: HParams) -> None:
    L.seed_everything(hparams.seed)

    if hparams.dataset == Dataset.CMNIST:
        data_module = cmnist
    elif hparams.dataset == Dataset.FUNK22:
        data_module = funk22
    else:
        assert hparams.dataset == Dataset.CELLPAINTING2
        data_module = cellpainting2

    data_train, data_val, data_all, metadata = data_module.get_data(hparams)
    
    hparams.img_channels = metadata["img_channels"]
    hparams.y_size = metadata["y_size"]
    hparams.e_size = metadata["e_size"]
    
    logger = CSVLogger(hparams.results_dir, name="", version=hparams.seed)
    
    if hparams.ckpt_path is None:
        model = Model(hparams)
    else:
        model = Model.load_from_checkpoint(hparams.ckpt_path, results_dir=hparams.results_dir)

    trainer = L.Trainer(
        logger=logger,
        callbacks=[ModelCheckpoint(filename="{step}", save_top_k=-1)],
        max_steps=hparams.num_train_steps,
        val_check_interval=hparams.num_steps_per_val,
        limit_val_batches=hparams.limit_val_batches,
        deterministic=True,
    )

    if hparams.is_embed:
        trainer.test(model, data_all)
    else:
        trainer.fit(model, data_train, data_val, ckpt_path=hparams.ckpt_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--dataset", type=Dataset, choices=list(Dataset), required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--img_pixels", type=int, default=64)
    parser.add_argument("--is_embed", action="store_true")
    
    # Model
    parser.add_argument("--encoder_type", type=EncoderType, choices=list(EncoderType), default=EncoderType.RESNET18)
    parser.add_argument("--z_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.)
    
    # Optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--y_per_batch", type=int, default=128)
    parser.add_argument("--num_train_steps", type=int, default=50000)
    parser.add_argument("--num_steps_per_val", type=int, default=5000)
    parser.add_argument("--limit_val_batches", type=int, default=500)
    
    hparams = HParams()
    hparams.update(parser.parse_args().__dict__)
    
    main(hparams)