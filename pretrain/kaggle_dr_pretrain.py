import os
import random

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata
import wandb

from pathlib import Path

from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn import model_selection


# =================================================
# Env #
# =================================================
rand = random.randint(0, 100000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =================================================
# Config #
# =================================================
conf = """
base:
    train_path: './data/KaggleDR/resized_train_cropped/resized_train_cropped'
    print_freq: 100
    num_workers: 20
    seed: 42
    target_size: 1
    target_cols: ['level']

    n_fold: 4
    trn_fold: [0]
    train: True
    debug: False
    oof: False

split:
    name: 'StratifiedKFold'
    param: {
        'n_splits': 4,
        'shuffle': True,
        'random_state': 1212
    }

model:
    model_name: 'tf_efficientnet_b0_ns'
    size: 320
    batch_size: 128
    pretrained: True
    epochs: 35

loss:
    name: 'MSELoss'
    param: {

    }

optimizer:
    name: 'AdamW'
    param: {
        'lr': 5e-3,
        'weight_decay': 1e-6,
        'amsgrad': False
    }

scheduler:
    name: 'CosineAnnealingLR'
    param: {
        'T_max': 10,
        'eta_min': 0,
        'last_epoch': -1
    }

wandb:
    use: True
    name: 'kaggle-dr-pretrain-simple'
    project: 'kaggle-dr-pretrain'
    tags: [
        'tf_efficientnet_b0_ns',
        'kaggle dr'
    ]
"""


# =================================================
# Utilities #
# =================================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================================================
# Transforms #
# =================================================
def get_transforms(img_size: int, mode="train"):
    if mode == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            A.IAAAdditiveGaussianNoise(),
            ToTensorV2()
        ])
    elif mode == "valid":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])
    else:
        raise NotImplementedError


# =================================================
# Dataset #
# =================================================
class TrainDataset(torchdata.Dataset):
    def __init__(self, cfg, df: pd.DataFrame, transform=None):
        self.df = df
        self.cfg = cfg
        self.filenames = df["image"].values
        self.labels = df["level"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        filepath = f"{self.cfg.base.train_path}/{filename}.jpeg"
        image = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        label = torch.tensor(self.labels[index]).float()
        return image, label


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 cfg,
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 fold_id: int = 0):
        super().__init__()

        self.cfg = cfg
        self.fold_id = fold_id

        self.train_df = train_df
        self.val_df = val_df

    def train_dataloader(self):
        train_dataset = TrainDataset(
            self.cfg,
            self.train_df,
            transform=get_transforms(self.cfg.model.size, mode="train"))
        return torchdata.DataLoader(
            train_dataset,
            batch_size=self.cfg.model.batch_size,
            num_workers=self.cfg.base.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True)

    def val_dataloader(self):
        valid_dataset = TrainDataset(
            self.cfg,
            self.val_df,
            transform=get_transforms(self.cfg.model.size, mode="valid"))
        return torchdata.DataLoader(
            valid_dataset,
            batch_size=self.cfg.model.batch_size,
            num_workers=self.cfg.base.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False)


# =================================================
# Train Utilities #
# =================================================
__CRITERIONS__: dict = {}
__OPTIMIZERS__: dict = {}
__SPLITS__: dict = {}


def get_criterion(cfg):
    if hasattr(nn, cfg.loss.name):
        return nn.__getattribute__(cfg.loss.name)(**cfg.loss.param)
    elif __CRITERIONS__.get(cfg.loss.name) is not None:
        return __CRITERIONS__[cfg.loss.name](**cfg.loss.param)
    else:
        raise NotImplementedError


def get_optimizer(cfg, model):
    optimizer_name = cfg.optimizer.name
    if hasattr(optim, optimizer_name):
        return optim.__getattribute__(optimizer_name)(model.parameters(), **cfg.optimizer.param)
    elif __OPTIMIZERS__.get(optimizer_name) is not None:
        return __OPTIMIZERS__[optimizer_name](model.parameters(), **cfg.optimizer.param)
    else:
        raise NotImplementedError


def get_scheduler(cfg, optimizer):
    scheduler_name = cfg.scheduler.name
    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(optimizer, **cfg.scheduler.param)


def get_split(cfg):
    if hasattr(model_selection, cfg.split.name):
        return model_selection.__getattribute__(cfg.split.name)(**cfg.split.param)
    elif __SPLITS__.get(cfg.split.name) is not None:
        return __SPLITS__[cfg.split.name](**cfg.split.param)
    else:
        raise NotImplementedError


# =================================================
# Model #
# =================================================
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.model = timm.create_model(cfg.model.model_name, pretrained=cfg.model.pretrained)
        if hasattr(self.model, "fc"):
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, cfg.base.target_size)
        elif hasattr(self.model, "classifier"):
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, cfg.base.target_size)
        else:
            raise NotImplementedError

        self.model.avg_pool = GeM()

        self.optimizer = get_optimizer(cfg, self.model)
        self.scheduler = get_scheduler(cfg, self.optimizer)
        self.criterion = get_criterion(cfg)

    def forward(self, x):
        return self.model(x).flatten()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).flatten()
        return self.criterion(y_hat, y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y).flatten()
        self.log("valid_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler

        return [optimizer], [scheduler]


# =================================================
# Train Loop #
# =================================================
def train_loop(cfg, folds, fold):
    global rand
    if cfg.wandb.use:
        wandb.init(
            cfg.wandb.name + f"-fold-{fold}-{rand}",
            project=cfg.wandb.project,
            tags=cfg.wandb.tags + [str(rand)],
            reinit=True)
        wandb_logger = WandbLogger(
            name=cfg.wandb.name + f"-fold-{fold}-{rand}",
            project=cfg.wandb.project,
            tags=cfg.wandb.tags + [str(rand)])
        wandb_logger.log_hyperparams(dict(cfg))
        wandb_logger.log_hyperparams({
            "rand": rand,
            "fold": fold,
        })
    trn_idx = folds[folds["fold"] != fold].index
    val_idx = folds[folds["fold"] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    data_module = DataModule(
        cfg,
        train_folds,
        valid_folds,
        fold_id=fold)
    model = Model(cfg)

    checkpoint_callback = ModelCheckpoint(
        filepath=Path("out") / __file__.__name__.split("/")[-1].replace(".py", "") / f"fold-{fold}",
        mode="min")

    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=cfg.model.epochs,
        gradient_clip_val=0.1,
        precision=16,
        logger=wandb_logger if "wandb_logger" in locals() else None,
        callbacks=[checkpoint_callback])

    trainer.fit(model=model, datamodule=data_module)


def main(cfg):
    train = pd.read_csv("./data/KaggleDR/trainLabels_cropped.csv")
    set_seed(seed=cfg.base.seed)

    folds = train.copy()
    kf = get_split(cfg)
    for n, (trn_idx, val_idx) in enumerate(kf.split(folds, folds[cfg.base.target_cols])):
        folds.loc[val_idx, "fold"] = int(n)

    for fold in range(cfg.base.n_fold):
        if fold in cfg.base.trn_fold:
            train_loop(cfg, folds, fold)


if __name__ == "__main__":
    main(OmegaConf.create(conf))
