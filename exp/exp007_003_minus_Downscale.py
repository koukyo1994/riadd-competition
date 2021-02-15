import gc
import os
import math
import random
import warnings

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from pathlib import Path
from typing import List

from albumentations.pytorch import ToTensorV2
from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.dl import Runner, SupervisedRunner
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn import model_selection
from sklearn import metrics
from torch.optim.optimizer import Optimizer
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
    base_model_name = "tf_efficientnet_b0_ns"
    pooling = "GeM"
    pretrained = True
    num_classes = 29

    ######################
    # Criterion #
    ######################
    loss_name = "BCEFocalLoss"
    loss_params: dict = {}

    ######################
    # Optimizer #
    ######################
    optimizer_name = "Adam"
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
        return model_selection.__getattribute__(CFG.split)(**CFG.scheduler_params)
    else:
        return MultilabelStratifiedKFold(**CFG.split_params)


# =================================================
# Dataset #
# =================================================
def crop_image_from_gray(image: np.ndarray, threshold: int = 7):
    if image.ndim == 2:
        mask = image > threshold
        return image[np.ix_(mask.any(1), mask.any(0))]
    elif image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray_image > threshold

    check_shape = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if (check_shape == 0):
        return image
    else:
        image1 = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        image2 = image[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        image3 = image[:, :, 2][np.ix_(mask.any(1), mask.any(0))]

        image = np.stack([image1, image2, image3], axis=-1)
        return image


class TrainDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path, target_columns: list, transform=None,
                 center_crop=False):
        self.df = df
        self.filenames = df["ID"].values
        self.datadir = datadir
        self.target_columns = target_columns
        self.labels = df[target_columns].values
        self.transform = transform
        self.center_crop = center_crop

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        path = self.datadir / f"{filename}.png"
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        if self.center_crop:
            image = crop_image_from_gray(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = torch.tensor(self.labels[index]).float()
        return {
            "ID": filename,
            "image": image,
            "targets": label
        }


class TestDataset(torchdata.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path, transform=None, center_crop=False):
        self.df = df
        self.filenames = df["ID"].values
        self.datadir = datadir
        self.transform = transform
        self.center_crop = center_crop

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        path = self.datadir / f"{filename}.png"
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        if self.center_crop:
            image = crop_image_from_gray(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return {
            "ID": filename,
            "image": image
        }


# =================================================
# Transforms #
# =================================================
def get_transforms(img_size: int, mode="train"):
    if mode == "train":
        return A.Compose([
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=(0.9, 1.1),
                ratio=(0.9, 1.1),
                p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=180,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=5,
                val_shift_limit=5,
                p=0.5),
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.4406],
                std=[0.229, 0.224, 0.225],
                always_apply=True),
            ToTensorV2()
        ])
    elif mode == "valid":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.4406],
                std=[0.229, 0.224, 0.225],
                always_apply=True),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.4406],
                std=[0.229, 0.224, 0.225],
                always_apply=True),
            ToTensorV2()
        ])


# =================================================
# Model #
# =================================================
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def gem(x: torch.Tensor, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class TimmModel(nn.Module):
    def __init__(self, base_model_name="tf_efficientnet_b0_ns", pooling="GeM", pretrained=True, num_classes=24):
        super().__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=pretrained)
        if hasattr(self.base_model, "fc"):
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.base_model, "classifier"):
            in_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise NotImplementedError

        if pooling == "GeM":
            self.base_model.avg_pool = GeM()

        self.init_layer()

    def init_layer(self):
        init_layer(self.base_model.classifier)

    def forward(self, x):
        return self.base_model(x)


# =================================================
# Optimizer and Scheduler #
# =================================================
version_higher = (torch.__version__ >= "1.5.0")


class AdaBelief(Optimizer):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: False) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: False) If set as True, then perform the rectified
            update similar to RAdam
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients
               NeurIPS 2020 Spotlight
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, weight_decouple=False, fixed_decay=False, rectify=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdaBelief, self).__init__(params, defaults)
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in AdaBelief')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in AdaBelief')
        if amsgrad:
            print('AMS enabled in AdaBelief')

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']
                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(
                    p.data,
                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(
                    p.data,
                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                state = self.state[p]
                beta1, beta2 = group['betas']
                # State initialization
                if len(state) == 0:
                    state['rho_inf'] = 2.0 / (1.0 - beta2) - 1.0
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(
                            p.data,
                            memory_format=torch.preserve_format) if version_higher else torch.zeros_like(p.data)
                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(group['weight_decay'], p.data)
                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(1 - beta2, grad_residual, grad_residual)
                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:  # Rectified update
                    # calculate rho_t
                    state['rho_t'] = state['rho_inf'] - 2 * state['step'] * beta2 ** state['step'] / (
                            1.0 - beta2 ** state['step'])
                    if state['rho_t'] > 4:  # perform Adam style update if variance is small
                        rho_inf, rho_t = state['rho_inf'], state['rho_t']
                        rt = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf / (rho_inf - 4.0) / (rho_inf - 2.0) / rho_t
                        rt = math.sqrt(rt)
                        step_size = rt * group['lr'] / bias_correction1
                        p.data.addcdiv_(-step_size, exp_avg, denom)
                    else:  # perform SGD style update
                        p.data.add_(-group['lr'], exp_avg)
        return loss


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
    "AdaBelief": AdaBelief,
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
        target = target.to(device)

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
    warnings.filterwarnings("ignore")

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

            loaders = {
                phase: torchdata.DataLoader(
                    TrainDataset(
                        df_,
                        CFG.train_datadir,
                        CFG.target_columns,
                        transform=get_transforms(CFG.img_size, phase)),
                    **CFG.loader_params[phase])  # type: ignore
                for phase, df_ in zip(["train", "valid"], [trn_df, val_df])
            }

            model = TimmModel(
                base_model_name=CFG.base_model_name,
                pooling=CFG.pooling,
                pretrained=CFG.pretrained,
                num_classes=CFG.num_classes)
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
            val_dataset = TrainDataset(
                val_df,
                CFG.train_datadir,
                CFG.target_columns,
                transform=get_transforms(CFG.img_size, "valid"))
            val_loader = torchdata.DataLoader(val_dataset, **CFG.loader_params["valid"])  # type: ignore

            model = TimmModel(
                base_model_name=CFG.base_model_name,
                pooling=CFG.pooling,
                pretrained=CFG.pretrained,
                num_classes=CFG.num_classes)
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
        test_dataset = TestDataset(test, CFG.test_datadir, transform=get_transforms(CFG.img_size, mode="test"))
        test_loader = torchdata.DataLoader(test_dataset, **CFG.loader_params["test"])  # type: ignore
        prediction_dfs = []
        for i in CFG.folds:
            logger.info("=" * 20)
            logger.info(f"Fold {i} inference")
            logger.info("=" * 20)
            model = TimmModel(
                base_model_name=CFG.base_model_name,
                pooling=CFG.pooling,
                pretrained=CFG.pretrained,
                num_classes=CFG.num_classes)
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
