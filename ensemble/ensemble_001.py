import gc
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as torchdata

from pathlib import Path
from typing import List

from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.dl import Runner, SupervisedRunner
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn import model_selection
from sklearn import metrics
from tqdm import tqdm


# =================================================
# Config #
# =================================================
class CFG:
    ######################
    # Globals #
    ######################
    seed = 1213
    epochs = 55
    train = True
    oof = True
    inference = True
    folds = [0, 1, 2, 3, 4]
    img_size = 320
    main_metric = "epoch_score"
    minimize_metric = False

    ######################
    # Data #
    ######################
    train_datadir = Path("data/Training_Set/Training")
    test_datadir = Path("data/Evaluation_Set")
    train_csv = "data/Training_Set/Camera_annotated.csv"
    test_csv = "data/Evaluate_camera_annotated.csv"

    ######################
    # Dataset #
    ######################
    target_columns = [
        "Disease_Risk", "DR", "ARMD", "MH", "DN",
        "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
        "CSR", "ODC", "CRVO", "TV", "AH", "ODP",
        "ODE", "ST", "AION", "PT", "RT", "RS", "CRS",
        "EDN", "RPEC", "MHL", "RP", "OTHER"
    ]

    ######################
    # Loaders #
    ######################
    loader_params = {
        "train": {
            "batch_size": 64,
            "num_workers": 20,
            "shuffle": True
        },
        "valid": {
            "batch_size": 64,
            "num_workers": 20,
            "shuffle": False
        },
        "test": {
            "batch_size": 64,
            "num_workers": 20,
            "shuffle": False
        }
    }

    ######################
    # Split #
    ######################
    split = "MultilabelStratifiedKFold"
    split_params = {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 1213
    }

    ######################
    # Model #
    ######################
    num_classes = 29

    ######################
    # Criterion #
    ######################
    loss_name = "BCEFocalLoss"
    loss_params: dict = {}

    ######################
    # Optimizer #
    ######################
    optimizer_name = "SAM"
    base_optimizer = "Adam"
    optimizer_params = {
        "lr": 0.001
    }
    # For SAM optimizer
    base_optimizer = "Adam"

    ######################
    # Scheduler #
    ######################
    scheduler_name = "CosineAnnealingLR"
    scheduler_params = {
        "T_max": 10
    }


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


def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def prepare_model_fore_inference(model, path: Path):
    if not torch.cuda.is_available():
        ckpt = torch.load(path, map_location="cpu")
    else:
        ckpt = torch.load(path)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# =================================================
# Split #
# =================================================
def get_split():
    if hasattr(model_selection, CFG.split):
        return model_selection.__getattribute__(CFG.split)(**CFG.split_params)
    else:
        return MultilabelStratifiedKFold(**CFG.split_params)


# =================================================
# Dataset #
# =================================================
class TrainDataset(torchdata.Dataset):
    def __init__(self, dfs: List[pd.DataFrame], target_df: pd.DataFrame, target_columns: list):
        self.dfs = dfs
        self.target_df = target_df
        self.target_columns = target_columns

    def __len__(self):
        return len(self.dfs[0])

    def __getitem__(self, index: int):
        predictions = []
        ID = self.dfs[0].loc[index, "ID"]
        for df in self.dfs:
            predictions.append(df.loc[index, self.target_columns].values.reshape(-1))
        preds = np.column_stack(predictions)
        preds_tensor = torch.tensor(preds).float().unsqueeze(0)  # (1, h, w)

        targets = self.target_df.loc[index, self.target_columns].values
        label = torch.tensor(targets).float()
        return {
            "ID": ID,
            "image": preds_tensor,
            "targets": label
        }


class TestDataset(torchdata.Dataset):
    def __init__(self, dfs: List[pd.DataFrame], target_columns: list):
        self.dfs = dfs
        self.target_columns = target_columns

    def __len__(self):
        return len(self.dfs[0])

    def __getitem__(self, index: int):
        predicions = []
        ID = self.dfs[0].loc[index, "ID"]
        for df in self.dfs:
            predicions.append(df.loc[index, self.target_columns].values.reshape(-1))
        preds = np.column_stack(predicions)
        preds_tensor = torch.tensor(preds).float().unsqueeze(0)

        return {
            "ID": ID,
            "image": preds_tensor
        }


# =================================================
# Model #
# =================================================
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


class StackingModel(nn.Module):
    def __init__(self, num_classes=29):
        self.convolution_layer = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=8,
                      kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=(1, 3)),
            nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(in_features=32 * num_classes,
                      out_features=64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        features = self.convolution_layer(x)
        features = features.mean(dim=-1).view(batch_size, -1)
        return self.classifier(features)


# =================================================
# Optimizer and Scheduler #
# =================================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm


__OPTIMIZERS__ = {
    "SAM": SAM,
}


def get_optimizer(model: nn.Module):
    optimizer_name = CFG.optimizer_name
    if optimizer_name == "SAM":
        base_optimizer_name = CFG.base_optimizer
        if __OPTIMIZERS__.get(base_optimizer_name) is not None:
            base_optimizer = __OPTIMIZERS__[base_optimizer_name]
        else:
            base_optimizer = optim.__getattribute__(base_optimizer_name)
        return SAM(model.parameters(), base_optimizer, **CFG.optimizer_params)

    if __OPTIMIZERS__.get(optimizer_name) is not None:
        return __OPTIMIZERS__[optimizer_name](model.parameters(),
                                              **CFG.optimizer_params)
    else:
        return optim.__getattribute__(optimizer_name)(model.parameters(),
                                                      **CFG.optimizer_params)


def get_scheduler(optimizer):
    scheduler_name = CFG.scheduler_name

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **CFG.scheduler_params)


# =================================================
# Criterion #
# =================================================
# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * (1. - probas)**self.gamma * bce_loss + (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss


__CRITERIONS__ = {
    "BCEFocalLoss": BCEFocalLoss,
}


def get_criterion():
    if hasattr(nn, CFG.loss_name):
        return nn.__getattribute__(CFG.loss_name)(**CFG.loss_params)
    elif __CRITERIONS__.get(CFG.loss_name) is not None:
        return __CRITERIONS__[CFG.loss_name](**CFG.loss_params)
    else:
        raise NotImplementedError


# =================================================
# Callbacks #
# =================================================
class SchedulerCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.Scheduler)

    def on_loader_end(self, state: IRunner):
        lr = state.scheduler.get_last_lr()
        state.epoch_metrics["lr"] = lr[0]
        if state.is_train_loader:
            state.scheduler.step()


class AUCCallback(Callback):
    def __init__(self, input_key: str = "targets", output_key: str = "logits", prefix="auc"):
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix

    def on_loader_start(self, runner: IRunner):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, runner: IRunner):
        targ = runner.input[self.input_key].detach().cpu().numpy()
        out = torch.sigmoid(runner.output[self.output_key].detach()).cpu().numpy()

        y_true = targ[:, 0].reshape(-1)
        y_pred = out[:, 0].reshape(-1)

        self.prediction.append(y_pred)
        self.target.append(y_true)

        score = metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
        runner.batch_metrics[self.prefix] = score

    def on_loader_end(self, runner: IRunner):
        y_pred = np.concatenate(self.prediction)
        y_true = np.concatenate(self.target)
        score = metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
        if runner.is_valid_loader:
            runner.epoch_metrics[runner.valid_loader + "_epoch_" + self.prefix] = score
        else:
            runner.epoch_metrics["train_epoch_" + self.prefix] = score


def multi_disease_avg_score(y_true: np.ndarray, y_pred: np.ndarray):
    map_score = metrics.average_precision_score(y_true=y_true, y_score=y_pred, average=None)
    map_score = np.nan_to_num(map_score, nan=0.0).mean()

    scores = []
    for i in range(len(y_true[0])):
        if y_true[:, i].mean() > 0.0:
            auc = metrics.roc_auc_score(y_true=y_true[:, i], y_score=y_pred[:, i])
            scores.append(auc)
        else:
            scores.append(0.0)
    auc_score = np.mean(scores)
    return 0.5 * map_score + 0.5 * auc_score


class MultiDiseaseAvgScore(Callback):
    def __init__(self, input_key: str = "targets", output_key: str = "logits", prefix="MdAS"):
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix

    def on_loader_start(self, runner: IRunner):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, runner: IRunner):
        targ = runner.input[self.input_key].detach().cpu().numpy()
        out = torch.sigmoid(runner.output[self.output_key].detach()).cpu().numpy()

        y_true = targ[:, 1:]
        y_pred = out[:, 1:]

        self.prediction.append(y_pred)
        self.target.append(y_true)

        score = multi_disease_avg_score(y_true=y_true, y_pred=y_pred)

        runner.batch_metrics[self.prefix] = score

    def on_loader_end(self, runner: IRunner):
        y_pred = np.concatenate(self.prediction)
        y_true = np.concatenate(self.target)

        score = multi_disease_avg_score(y_true, y_pred)
        if runner.is_valid_loader:
            runner.epoch_metrics[runner.valid_loader + "_epoch_" + self.prefix] = score
        else:
            runner.epoch_metrics["train_epoch_" + self.prefix] = score


class CompetitionScore(Callback):
    def __init__(self, input_key: str = "targets", output_key: str = "logits", prefix="score"):
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix

    def on_loader_start(self, runner: IRunner):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, runner: IRunner):
        targ = runner.input[self.input_key].detach().cpu().numpy()
        out = torch.sigmoid(runner.output[self.output_key].detach()).cpu().numpy()

        self.prediction.append(out)
        self.target.append(targ)

        y_true_auc = targ[:, 0].reshape(-1)
        y_pred_auc = out[:, 0].reshape(-1)

        y_true_mdas = targ[:, 1:]
        y_pred_mdas = out[:, 1:]

        auc_score = metrics.roc_auc_score(y_true=y_true_auc, y_score=y_pred_auc)

        mdas = multi_disease_avg_score(y_true_mdas, y_pred_mdas)

        score = 0.5 * auc_score + 0.5 * mdas

        runner.batch_metrics[self.prefix] = score

    def on_loader_end(self, runner: IRunner):
        y_pred = np.concatenate(self.prediction)
        y_true = np.concatenate(self.target)

        y_true_auc = y_true[:, 0].reshape(-1)
        y_pred_auc = y_pred[:, 0].reshape(-1)

        y_true_mdas = y_true[:, 1:]
        y_pred_mdas = y_pred[:, 1:]

        auc_score = metrics.roc_auc_score(y_true=y_true_auc, y_score=y_pred_auc)

        mdas_score = multi_disease_avg_score(y_true_mdas, y_pred_mdas)
        score = 0.5 * auc_score + 0.5 * mdas_score
        if runner.is_valid_loader:
            runner.epoch_metrics[runner.valid_loader + "_epoch_" + self.prefix] = score
        else:
            runner.epoch_metrics["train_epoch_" + self.prefix] = score


def get_callbacks():
    if CFG.optimizer_name == "SAM":
        return [
            SchedulerCallback(),
            AUCCallback(),
            MultiDiseaseAvgScore(),
            CompetitionScore()
        ]
    else:
        return [
            AUCCallback(),
            MultiDiseaseAvgScore(),
            CompetitionScore()
        ]


# =================================================
# Runner #
# =================================================
class SAMRunner(Runner):
    def predict_batch(self, batch, **kwargs):
        return super().predict_batch(batch, **kwargs)

    def _handle_batch(self, batch):
        input_, target = batch["image"], batch["targets"]

        input_ = input_.to(self.device)
        target = target.to(self.device)

        out = self.model(input_)

        loss = self.criterion(out, target)
        self.batch_metrics.update({
            "loss": loss
        })

        self.input = batch
        self.output = {"logits": out}

        if self.is_train_loader:
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            self.criterion(self.model(input_), target).backward()
            self.optimizer.second_step(zero_grad=True)


def get_runner(device: torch.device):
    if CFG.optimizer_name == "SAM":
        return SAMRunner(device=device)
    else:
        return SupervisedRunner(device=device, input_key="image", input_target_key="targets")


if __name__ == "__main__":
    # logging
    filename = __file__.split("/")[-1].replace(".py", "")
    logdir = Path(f"out/{filename}")
    logdir.mkdir(exist_ok=True, parents=True)

    # environment
    set_seed(CFG.seed)
    device = get_device()

    # validation
    splitter = get_split()

    # data
    train = pd.read_csv(CFG.train_csv)
    test = pd.read_csv(CFG.test_csv)

    # predictions
    candidates = [
        Path("out/exp012_SAM"), Path("out/exp014_EfficientNetB1"),
        Path("out/exp015_SAM_with_large_image_size"),
        Path("out/exp022_015_amp_plus_larger_batch_size"),
        Path("out/exp023_012_better_crop"),
        Path("out/exp026_larger_image_size2"),
        Path("out/exp027_nfnet"),
        Path("out/exp030_EfficientNetB3")
    ]

    oofs = []
    submissions = []
    names = []
    for cand in candidates:
        oofs.append(pd.read_csv(cand / "oof.csv"))
        submissions.append(pd.read_csv(cand / "chizu & arai & okada_results.csv"))
        names.append(cand.name)

    columns = train.columns
    for i in range(len(oofs)):
        oofs[i] = oofs[i][columns].sort_values(by="ID").reset_index(drop=True)

    if CFG.train:
        logger = init_logger(log_file=logdir / "train.log")
        for i, (trn_idx, val_idx) in enumerate(splitter.split(train, y=train[CFG.target_columns])):
            if i not in CFG.folds:
                continue

            logger.info("=" * 20)
            logger.info(f"Fold {i} Training")
            logger.info("=" * 20)

            trn_df = train.loc[trn_idx, :].reset_index(drop=True)
            val_df = train.loc[val_idx, :].reset_index(drop=True)

            trn_oofs = [oof.loc[trn_idx, :].reset_index(drop=True) for oof in oofs]
            val_oofs = [oof.loc[val_idx, :].reset_index(drop=True) for oof in oofs]

            trn_dataset = TrainDataset(trn_oofs, trn_df, CFG.target_columns)
            val_dataset = TrainDataset(val_oofs, val_df, CFG.target_columns)

            loaders = {
                phase: torchdata.DataLoader(
                    dataset,
                    **CFG.loader_params[phase])  # type: ignore
                for phase, dataset in zip(["train", "valid"], [trn_dataset, val_dataset])
            }

            model = StackingModel(num_classes=CFG.num_classes)
            criterion = get_criterion()
            optimizer = get_optimizer(model)
            scheduler = get_scheduler(optimizer)
            callbacks = get_callbacks()
            runner = get_runner(device)
            runner.train(
                model=model,
                criterion=criterion,
                loaders=loaders,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=CFG.epochs,
                verbose=True,
                logdir=logdir / f"fold{i}",
                callbacks=callbacks,
                main_metric=CFG.main_metric,
                minimize_metric=CFG.minimize_metric)

            del model, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()

    if CFG.oof:
        logger = init_logger(log_file=logdir / "oof.log")
        prediction_dfs = []
        targets_dfs = []
        for i, (_, val_idx) in enumerate(splitter.split(train, y=train[CFG.target_columns])):
            if i not in CFG.folds:
                continue
            logger.info("=" * 20)
            logger.info(f"Fold {i} OOF")
            logger.info("=" * 20)

            val_df = train.loc[val_idx, :].reset_index(drop=True)
            val_oofs = [oof.loc[val_idx, :].reset_index(drop=True) for oof in oofs]
            val_dataset = TrainDataset(val_oofs, val_df, CFG.target_columns)
            val_loader = torchdata.DataLoader(val_dataset, **CFG.loader_params["valid"])  # type: ignore

            model = StackingModel(num_classes=CFG.num_classes)
            model = prepare_model_fore_inference(model, logdir / f"fold{i}/checkpoints/best.pth").to(device)
            predictions = []
            targets = []
            ids = []
            for batch in tqdm(val_loader, desc=f"fold{i} oof"):
                input_ = batch["image"]
                label = batch["targets"]
                id_ = batch["ID"]
                input_ = input_.to(device)
                targets.append(label.cpu().numpy())
                ids.extend(id_.cpu().numpy().tolist())
                with torch.no_grad():
                    output = model(input_).detach()
                predictions.append(torch.sigmoid(output).cpu().numpy())

            pred_array = np.concatenate(predictions, axis=0)
            pred_df = pd.DataFrame(pred_array, columns=CFG.target_columns)
            pred_df["ID"] = ids

            targ_array = np.concatenate(targets, axis=0)
            targ_df = pd.DataFrame(targ_array, columns=CFG.target_columns)
            targ_df["ID"] = ids

            prediction_dfs.append(pred_df)
            targets_dfs.append(targ_df)

            y_true_auc = targ_array[:, 0].reshape(-1)
            y_pred_auc = pred_array[:, 0].reshape(-1)
            auc = metrics.roc_auc_score(y_true=y_true_auc, y_score=y_pred_auc)
            logger.info(f"Fold {i} AUC: {auc:.5f}")

            y_true_mdas = targ_array[:, 1:]
            y_pred_mdas = pred_array[:, 1:]
            mdas_score = multi_disease_avg_score(y_true_mdas, y_pred_mdas)
            logger.info(f"Fold {i} Multi-disease Avg Score: {mdas_score:.5g}")

            score = 0.5 * auc + 0.5 * mdas_score
            logger.info(f"Fold {i} Final score: {score:.5f}")

        pred_df = pd.concat(prediction_dfs, axis=0).reset_index(drop=True)
        targ_df = pd.concat(targets_dfs, axis=0).reset_index(drop=True)

        y_true_auc = targ_df[CFG.target_columns[0]].values.reshape(-1)
        y_pred_auc = pred_df[CFG.target_columns[0]].values.reshape(-1)
        auc = metrics.roc_auc_score(y_true=y_true_auc, y_score=y_pred_auc)
        logger.info(f"AUC: {auc:.5f}")

        y_true_mdas = targ_df[CFG.target_columns[1:]].values
        y_pred_mdas = pred_df[CFG.target_columns[1:]].values
        mdas_score = multi_disease_avg_score(y_true_mdas, y_pred_mdas)
        logger.info(f"Multi-disease Avg Score: {mdas_score:.5g}")

        score = 0.5 * auc + 0.5 * mdas_score
        logger.info(f"Final score: {score:.5f}")

        pred_df.to_csv(logdir / "oof.csv", index=False)

    if CFG.inference:
        logger = init_logger(log_file=logdir / "inference.log")
        test_dataset = TestDataset(submissions, CFG.target_columns)
        test_loader = torchdata.DataLoader(test_dataset, **CFG.loader_params["test"])  # type: ignore
        prediction_dfs = []
        for i in CFG.folds:
            logger.info("=" * 20)
            logger.info(f"Fold {i} inference")
            logger.info("=" * 20)
            model = StackingModel(num_classes=CFG.num_classes)
            model = prepare_model_fore_inference(model, logdir / f"fold{i}/checkpoints/best.pth").to(device)
            predictions = []
            ids = []
            for batch in tqdm(test_loader, desc=f"fold{i} inference"):
                input_ = batch["image"].to(device)
                id_ = batch["ID"]
                ids.extend(id_.cpu().numpy().tolist())
                with torch.no_grad():
                    output = model(input_).detach()
                predictions.append(torch.sigmoid(output).cpu().numpy())

            pred_array = np.concatenate(predictions, axis=0)
            pred_df = pd.DataFrame({
                "ID": ids
            })
            pred_df = pd.concat([pred_df, pd.DataFrame(pred_array, columns=CFG.target_columns)], axis=1)
            prediction_dfs.append(pred_df)

        submission_df = pd.DataFrame()
        submission_df["ID"] = prediction_dfs[0]["ID"]
        for column in CFG.target_columns:
            submission_df[column] = 0.0

        for df in prediction_dfs:
            submission_df[CFG.target_columns] += df[CFG.target_columns] / len(prediction_dfs)

        submission_df.to_csv(logdir / "chizu & arai & okada_results.csv", index=False)
