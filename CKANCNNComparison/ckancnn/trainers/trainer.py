from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..metrics.calibration import ece_score, nll_loss
from ..utils.logging import AverageMeter, CSVLogger


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    label_smoothing: float = 0.0
    grad_clip_norm: Optional[float] = None
    mc_dropout: bool = False
    mc_samples: int = 20
    mixed_precision: bool = False


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_classes: int,
        log_dir: str,
        config: TrainConfig,
        run_meta: Optional[Dict] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.cfg = config
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)
        # use torch.amp api on cuda mp only
        if self.cfg.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')  # type: ignore[arg-type]
        else:
            self.scaler = None
        self.logger = CSVLogger(log_dir)
        self.run_meta = run_meta or {}

    def _optimizer(self, lr: float):
        return optim.SGD(self.model.parameters(), lr=lr, momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay, nesterov=True)

    def fit(self, train_loader, val_loader, initial_lr: float = None):
        lr = self.cfg.lr if initial_lr is None else initial_lr
        optimizer = self._optimizer(lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.epochs)

        best_acc = 0.0
        for epoch in range(1, self.cfg.epochs + 1):
            train_stats = self._train_one_epoch(train_loader, optimizer)
            val_stats = self.evaluate(val_loader)
            scheduler.step()
            self.logger.log_epoch(epoch, train_stats, val_stats)
            if val_stats["acc"] > best_acc:
                best_acc = val_stats["acc"]
                self.logger.save_checkpoint(self.model, "best.pt", extra=self.run_meta)
            self.logger.save_checkpoint(self.model, "latest.pt", extra=self.run_meta)

    def _train_one_epoch(self, loader, optimizer):
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        pbar = tqdm(loader, desc="train", leave=False)
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip_norm is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                loss.backward()
                if self.cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                optimizer.step()

            preds = logits.argmax(dim=1)
            correct = (preds == targets).sum().item()
            acc = correct / targets.size(0)

            loss_meter.update(loss.item(), n=targets.size(0))
            acc_meter.update(acc, n=targets.size(0))
            pbar.set_postfix({"loss": f"{loss_meter.avg:.3f}", "acc": f"{acc_meter.avg:.3f}"})

        return {"loss": loss_meter.avg, "acc": acc_meter.avg}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        ece_meter = AverageMeter()
        nll_meter = AverageMeter()

        pbar = tqdm(loader, desc="eval", leave=False)
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            if self.cfg.mc_dropout:
                logits_mc = []
                # enable dropout during eval via train mode on dropout modules
                def _set_dropout(m):
                    if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                        m.train()

                self.model.apply(_set_dropout)
                for _ in range(self.cfg.mc_samples):
                    logits_mc.append(self.model(images))
                logits = torch.stack(logits_mc, dim=0).mean(dim=0)
                # restore eval mode after mc sampling
                self.model.eval()
            else:
                logits = self.model(images)

            probs = torch.softmax(logits, dim=1)
            loss = self.criterion(logits, targets)
            nll_val = nll_loss(probs, targets, eps=1e-12)
            ece_val = ece_score(probs, targets, n_bins=15)

            preds = logits.argmax(dim=1)
            correct = (preds == targets).sum().item()
            acc = correct / targets.size(0)

            loss_meter.update(loss.item(), n=targets.size(0))
            acc_meter.update(acc, n=targets.size(0))
            ece_meter.update(float(ece_val), n=targets.size(0))
            nll_meter.update(float(nll_val), n=targets.size(0))
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.3f}",
                "acc": f"{acc_meter.avg:.3f}",
                "ece": f"{ece_meter.avg:.3f}",
            })

        return {"loss": loss_meter.avg, "acc": acc_meter.avg, "ece": ece_meter.avg, "nll": nll_meter.avg}
