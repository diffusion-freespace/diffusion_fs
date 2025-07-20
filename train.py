import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from datasets_all.fs_cache_dataset import FSCacheDataset
from datasets_all.nusc_dataset import NuScenesDataset
from fs_model import LITFSModel
import yaml
import numpy as np

L.seed_everything(2024)
class Config:
    def __init__(self, config):
        self.config = config
        for k, v in self.config.items():
            self.__setattr__(k,  v)

config = "configs/carla.yaml"
# config = "configs/nuscenes.yaml"

with open(config, "r") as f:
    config = Config(yaml.safe_load(f))

dataset_class = {"nuscenes" : NuScenesDataset, "carla" : FSCacheDataset}

train_dataset = dataset_class[config.dataset_type](config.dataset_config)
val_dataset = dataset_class[config.dataset_type](config.dataset_config, is_train=False)

trainloader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, shuffle=False)
valloader = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, shuffle=False)

model = LITFSModel(config)

run_name = f"{config.dataset_type}_contour_{config.backbone}_{config.conditioning}"
# logger = WandbLogger(name=run_name, project="fs_carla_diff", config=config)
logger = CSVLogger("logs", name="diff_fs")

checkpoint_callback = ModelCheckpoint(monitor="val_iou", dirpath=config.ckpt_dir, filename=f"best_loss_{run_name}", mode='max')
checkpoint_callback2 = ModelCheckpoint(dirpath=config.ckpt_dir, filename=f'last_{run_name}', save_last=True)

trainer = L.Trainer(devices="auto", max_epochs=50, callbacks=[checkpoint_callback, checkpoint_callback2], logger=[logger],\
    strategy='ddp_find_unused_parameters_true', detect_anomaly=True)

trainer.fit(model, trainloader, valloader)