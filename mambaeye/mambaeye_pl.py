import torch
from transformers import get_cosine_schedule_with_warmup
import lightning as L
from mambaeye.loss import MambaEyeLoss
from mambaeye.model import MambaEye
from mambaeye.dataset import (
    collate_fn_keep_batch_size,
    ImagenetDatasetSinusoidal,
)

class MambaEyePL(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = MambaEye(**self.config.model)
        self.loss_fn = MambaEyeLoss()
        self.save_hyperparameters()

    def forward(
        self,
        img_sequence,
        move_embedding,
        inference_params=None,
    ):
        classification_sequence = self.model(
            img_sequence,
            move_embedding,
            inference_params=inference_params,
        )
        return classification_sequence

    def common_step(self, batch, batch_idx, train=True, inference_params=None):
        img_sequence, move_embedding, information_ratio, labels, absolute_location = (
            batch
        )
        classification_sequence = self(
            img_sequence, move_embedding, inference_params=inference_params
        )

        loss = self.loss_fn(
            classification_sequence,
            labels,
            information_ratio,
        )
        return (
            loss,
            classification_sequence,
        )

    def training_step(self, batch, batch_idx):
        loss, classification_sequence = self.common_step(
            batch, batch_idx, train=True
        )

        self.log("train/loss", loss)
        self.log("lr", self.optimizers().param_groups[0]["lr"])

        return loss

    def validation_step(self, batch, batch_idx):
        (
            loss,
            classification_sequence,
        ) = self.common_step(batch, batch_idx, train=False)
        self.log("val/loss", loss, sync_dist=True)

        # validation accuracy
        validation_length = classification_sequence.shape[1]
        labels = batch[3]
        for vlen in [256, 512, 1024, 2048, 4096]:
            if validation_length >= vlen:
                _, preds = torch.max(classification_sequence[:, vlen - 1, :], dim=-1)
                acc = torch.sum(preds == labels).item() / len(labels)
                self.log(f"val/acc_{vlen}", acc, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.config.optimizer)
        if "scheduler" in self.config:
            num_total_step = int(
                self.config.scheduler.steps_per_epoch * self.config.scheduler.total_epochs
            )
            num_warmup_steps = int(
                self.config.scheduler.steps_per_epoch * self.config.scheduler.warmup_epochs
            )
            num_training_steps = num_total_step
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return optimizer

    def train_dataloader(self):
        train_dataset = ImagenetDatasetSinusoidal(**self.config.dataset.train)
        return torch.utils.data.DataLoader(
            train_dataset,
            **self.config.dataloader.train,
            collate_fn=collate_fn_keep_batch_size,
        )

    def val_dataloader(self):
        val_dataset = ImagenetDatasetSinusoidal(**self.config.dataset.val)
        return torch.utils.data.DataLoader(val_dataset, **self.config.dataloader.val)
