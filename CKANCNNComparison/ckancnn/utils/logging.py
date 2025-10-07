import csv
import os
from typing import Dict

import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n


class CSVLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "phase", "loss", "acc", "ece", "nll"])

    def log_epoch(self, epoch: int, train_stats: Dict[str, float], val_stats: Dict[str, float]):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, "train", train_stats.get("loss", 0.0), train_stats.get("acc", 0.0), train_stats.get("ece", 0.0), train_stats.get("nll", 0.0)])
            writer.writerow([epoch, "val", val_stats.get("loss", 0.0), val_stats.get("acc", 0.0), val_stats.get("ece", 0.0), val_stats.get("nll", 0.0)])

    def save_checkpoint(self, model: torch.nn.Module, name: str, extra: dict | None = None):
        ckpt_dir = os.path.join(self.log_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, name)
        payload = {"state_dict": model.state_dict()}
        if extra:
            payload["meta"] = extra
        torch.save(payload, path)
