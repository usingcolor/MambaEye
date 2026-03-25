import os
import hashlib
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger

from mambaeye.mambaeye_pl import MambaEyePL

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Initialize the model
    model = MambaEyePL(cfg)

    # Setup Logger (make WandB optional gracefully)
    experiment_id = cfg.get("experiment_id", None)
    if cfg.wandb is not None and cfg.wandb.get("entity") is not None:
        logger = WandbLogger(
            entity=cfg.wandb.get("entity"),
            project=cfg.wandb.get("project", "MambaEye"),
            id=experiment_id,
        )
        logger.watch(model)
        experiment_id = logger.experiment.id
    else:
        logger = CSVLogger(save_dir="logs/", name="MambaEye")
        if experiment_id is None:
            # Generate a unique hash based on configuration to mimic previous behavior
            config_hash = hashlib.md5(OmegaConf.to_yaml(cfg).encode('utf-8')).hexdigest()
            experiment_id = "run_" + config_hash[:16]

    fine_tuning = cfg.get("fine_tuning", False)
    ckpt_path = cfg.get("ckpt_path", None)

    if fine_tuning:
        if ckpt_path is None:
            raise ValueError("Please provide a checkpoint path for fine-tuning via cfg.ckpt_path.")
        else:
            # lighting checkpoint
            if ckpt_path.endswith(".ckpt"):
                trained_model = MambaEyePL.load_from_checkpoint(ckpt_path)
                model.model.load_state_dict(trained_model.model.state_dict())
                del trained_model
            # pytorch checkpoint
            elif ckpt_path.endswith(".pt"):
                state_dict = torch.load(ckpt_path, map_location="cpu")
                model.model.load_state_dict(state_dict)
            else:
                raise ValueError("Please provide a valid checkpoint path ending in .ckpt or .pt.")
            ckpt_path = None

    # Initialize the trainer
    trainer = L.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        callbacks=[
            ModelCheckpoint(
                monitor="val/loss",
                dirpath="checkpoints/",
                filename=f"{experiment_id}_" + "{epoch}",
                save_top_k=-1,
            ),
        ],
        logger=logger,
    )

    # Train the model
    trainer.fit(model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
