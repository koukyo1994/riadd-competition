import codecs
import json
import os
import random

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Union

from sklearn import metrics
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


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


def multi_disease_avg_score_per_column(y_true: np.ndarray, y_pred: np.ndarray, return_each=False):
    map_score = metrics.average_precision_score(y_true=y_true, y_score=y_pred)
    auc = metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
    if return_each:
        return 0.5 * map_score + 0.5 * auc, map_score, auc
    else:
        return 0.5 * map_score + 0.5 * auc


def search_averaging_weights_per_column(predictions: list, target: np.ndarray, func, trials=10000):
    best_score = -np.inf
    best_weights = np.zeros(len(predictions))
    set_seed(1213)

    for i in tqdm(range(trials)):
        dice = np.random.rand(len(predictions))
        weights = dice / dice.sum()
        blended = np.zeros(len(predictions[0]))
        for weight, pred in zip(weights, predictions):
            blended += weight * pred
        score = func(target, blended)
        if score > best_score:
            best_score = score
            best_weights = weights
    return {"best_score": best_score, "best_weights": best_weights}


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def save_json(config: dict, save_path: Union[str, Path]):
    f = codecs.open(str(save_path), mode="w", encoding="utf-8")
    json.dump(config, f, indent=4, cls=MyEncoder, ensure_ascii=False)
    f.close()


if __name__ == "__main__":
    # logging
    filename = __file__.split("/")[-1].replace(".py", "")
    logdir = Path(f"out/{filename}")
    logdir.mkdir(exist_ok=True, parents=True)

    # ensemble candidates
    candidates = [
        Path("out/exp012_SAM"), Path("out/exp014_EfficientNetB1"),
        Path("out/exp015_SAM_with_large_image_size"),
        Path("out/exp022_015_amp_plus_larger_batch_size"),
        Path("out/exp023_012_better_crop"),
        Path("out/exp026_larger_image_size2"),
        Path("out/exp027_nfnet"),
        Path("out/exp030_EfficientNetB3"),
        Path("out/exp038_EfficientNetB3_384"),
        Path("out/exp039_EfficientNetB5_456"),
        Path("out/exp040_EfficientNetB4_384")
    ]

    oofs = []
    submissions = []
    names = []
    for cand in candidates:
        oofs.append(pd.read_csv(cand / "oof.csv"))
        submissions.append(pd.read_csv(cand / "chizu & arai & okada_results.csv"))
        names.append(cand.name)

    train = pd.read_csv("data/Training_Set/RFMiD_Training_Labels.csv")
    columns = train.columns
    for i in range(len(oofs)):
        oofs[i] = oofs[i][columns].sort_values(by="ID").reset_index(drop=True)

    weights_dict = {}
    classes = [
        "Disease_Risk", "DR", "ARMD", "MH", "DN",
        "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
        "CSR", "ODC", "CRVO", "TV", "AH", "ODP",
        "ODE", "ST", "AION", "PT", "RT", "RS", "CRS",
        "EDN", "RPEC", "MHL", "RP", "OTHER"
    ]

    for class_ in classes:
        predictions = []
        for oof in oofs:
            predictions.append(oof[class_].values)
        target = train[class_].values
        if class_ == "Disease_Risk":
            result_dict = search_averaging_weights_per_column(predictions, target, metrics.roc_auc_score)
        else:
            result_dict = search_averaging_weights_per_column(predictions, target, multi_disease_avg_score_per_column)
        weights_dict[class_] = result_dict["best_weights"]

    blended = np.zeros((len(oofs[0]), len(classes)))
    for class_ in weights_dict:
        index = classes.index(class_)
        weights = weights_dict[class_]
        for weight, oof in zip(weights, oofs):
            blended[:, index] += weight * oof[class_].values

    class_level_score = {}
    auc = metrics.roc_auc_score(train["Disease_Risk"].values, blended[:, 0])
    disease_classes = classes.copy()
    disease_classes.remove("Disease_Risk")
    mdas = multi_disease_avg_score(train[disease_classes].values, blended[:, 1:])
    class_level_score["Disease_Risk_AUC"] = auc
    class_level_score["Multi-disease-Avg-Score"] = mdas
    class_level_score["Competition-Score"] = 0.5 * auc + 0.5 * mdas
    class_level_score["blend_weight"] = weights_dict

    class_level_score["per_disease_score"] = {}
    for class_ in disease_classes:
        per_class_score, pc_map, pc_auc = multi_disease_avg_score_per_column(
            train[class_].values, blended[:, classes.index(class_)], return_each=True)
        class_level_score["per_disease_score"][class_] = {
            "Disease_score": per_class_score,
            "mAP": pc_map,
            "AUC": pc_auc
        }

    for oof, name in zip(oofs, names):
        auc = metrics.roc_auc_score(train["Disease_Risk"].values, oof["Disease_Risk"].values)
        mdas = multi_disease_avg_score(train[disease_classes].values, oof[disease_classes].values)
        class_level_score[f"{name}_AUC"] = auc
        class_level_score[f"{name}_Multi-disease-Avg-Score"] = mdas
        class_level_score[f"{name}_Competition-Score"] = 0.5 * auc + 0.5 * mdas
        class_level_score[f"{name}_per_disease_score"] = {}
        for class_ in disease_classes:
            per_class_score, pc_map, pc_auc = multi_disease_avg_score_per_column(
                train[class_].values, oof[class_].values, return_each=True)
            class_level_score[f"{name}_per_disease_score"][class_] = {
                "Disease_score": per_class_score,
                "mAP": pc_map,
                "AUC": pc_auc
            }

    blended_sub = np.zeros((len(submissions[0]), len(classes)))
    for class_ in weights_dict:
        index = classes.index(class_)
        for weight, sub in zip(weights_dict[class_], submissions):
            blended_sub[:, index] += weight * sub[class_].values

    sub = pd.concat([
        pd.DataFrame({"ID": submissions[0]["ID"]}),
        pd.DataFrame(blended_sub, columns=classes)
    ], axis=1)
    sub.to_csv(logdir / "chizu & arai & okada_results.csv", index=False)
    save_json(class_level_score, logdir / "class_level_results.json")
