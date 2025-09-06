import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("wine")


def load_pt(filepath):
    if os.path.exists(filepath):
        return torch.load(filepath)
    else:
        raise FileNotFoundError(filepath)


def train_val_split(X, y, n_train=3500, shuffle=True, seed=42):
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        idx = torch.randperm(X.size(0), generator=generator)
        X, y = X[idx], y[idx]
    return (X[:n_train], y[:n_train]), (X[n_train:], y[n_train:])


def standardize_fit(X):
    mean = X.mean(0)
    std = X.std(0)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return mean, std


def standardize_apply(X, mean, std):
    return (X - mean) / std


class SingleLayerWineModel(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, p_drop=0.0):
        super().__init__()

        self.layer1 = nn.Linear(in_dim, hidden)
        self.layer_out = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(p_drop)

        #self._init_kaiming()

    def _init_kaiming(self):
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity="relu")
        nn.init.zeros_(self.layer1.bias)
        nn.init.kaiming_normal_(self.layer_out.weight, nonlinearity="linear")
        nn.init.zeros_(self.layer_out.bias)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.drop(x)
        return self.layer_out(x)


@dataclass
class TrainConfig:
    task: str = 'regression'
    epoch: int = 1200
    batch_size: int = 128
    lr: float = 0.05
    weight_decay: float = 1e-4
    hidden: int = 5
    p_drop: float = 0.5


def train_model(X_train, y_train, X_val, y_val, cfg: TrainConfig):

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(X_val, y_val), batch_size=512, shuffle=False)

    if cfg.task == 'regression':
        out_dim = 1
    else:
        out_dim = 10

    model = SingleLayerWineModel(in_dim=X_train.shape[1], hidden=cfg.hidden, out_dim=out_dim, p_drop=cfg.p_drop)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(1, cfg.epoch + 1):
        model.train()
        epoch_sum, seen = 0.0, 0

        for xb, yb in train_dl:
            if cfg.task == 'regression':
                preds = model(xb)
                loss = F.mse_loss(preds, yb, reduction='mean')
            else:
                logits = model(xb)
                y_idx = (yb.view(-1).long() - 1).clamp_(0, 9)
                loss = F.cross_entropy(logits, y_idx, reduction='mean')

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_sum += loss.item() * xb.size(0)
            seen += xb.size(0)

        if epoch == 1 or epoch == cfg.epoch or epoch % 25 == 0:
            train_loss = epoch_sum / max(1, seen)
            if cfg.task == 'regression':
                val_mse = evaluate_regression(model, val_dl)
                log.info(f"[{cfg.task}] Epoch: {epoch}, train_MSE: {train_loss:.6f}, val_mse: {val_mse:.6f}")
            elif cfg.task == 'classification':
                val_ce, acc = evaluate_classification(model, val_dl)
                log.info(f"[{cfg.task}] Epoch: {epoch}, train_CE: {train_loss:.6f}, val_CE: {val_ce:.6f}, val_acc: {acc:.4f}")

    return model


@torch.no_grad()
def evaluate_regression(model, dl):
    model.eval()
    total_loss, seen = 0.0, 0
    for xb, yb in dl:
        preds = model(xb)
        loss = F.mse_loss(preds, yb, reduction='mean')
        total_loss += loss.item() * xb.size(0)
        seen += xb.size(0)

    return total_loss / max(1, seen)


@torch.no_grad()
def evaluate_classification(model, dl):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in dl:
        logits = model(xb)
        y_idx = (yb.view(-1).long() - 1).clamp_(0, 9)
        loss = F.cross_entropy(logits, y_idx, reduction='mean')
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_idx).sum().item()
        total += xb.size(0)
    ce = total_loss / max(1, total)
    acc = correct / max(1, total)

    return ce, acc


if __name__ == "__main__":
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="wine-white.pt", help="path to data")
    parser.add_argument("--task", choices=["regression", "classification"], default="regression",
                        help="task type, regression or classification")
    parser.add_argument("--epochs", type=int, default=1200, help="epoch number")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=5.5e-2, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2-regulization")
    parser.add_argument("--hidden", type=int, default=5, help="units in hidden layer")
    parser.add_argument("--p_drop", type=float, default=0.3, help="dropout value")
    args = parser.parse_args()

    data = load_pt(args.file)
    log.info(f"data size: {data.shape}")

    X, y = data[:, :-1].float(), data[:, -1:].float()

    (Xtr, ytr), (Xva, yva) = train_val_split(X, y, n_train=3500)

    mean, std = standardize_fit(Xtr)
    Xtr = standardize_apply(Xtr, mean, std)
    Xva = standardize_apply(Xva, mean, std)

    cfg = TrainConfig(task=args.task,
                      epoch=args.epochs,
                      batch_size=args.batch_size,
                      lr=args.lr,
                      weight_decay=args.weight_decay,
                      hidden=args.hidden,
                      p_drop=args.p_drop)

    model = train_model(Xtr, ytr, Xva, yva, cfg)
