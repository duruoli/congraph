import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

from builder import (
    ACTION_MODALITIES,
    ID2ACTION,
    build_il_dataset,
    patient_level_split,
    make_class_weights,
)

class PairDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    total = 0
    top1_correct = 0
    top3_correct = 0
    class_tp = np.zeros(n_classes, dtype=np.int64)
    class_tot = np.zeros(n_classes, dtype=np.int64)

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)

        pred1 = logits.argmax(dim=1)
        top1_correct += (pred1 == yb).sum().item()

        k = min(3, n_classes)
        topk_idx = torch.topk(logits, k=k, dim=1).indices
        top3_correct += (topk_idx == yb.unsqueeze(1)).any(dim=1).sum().item()

        y_np = yb.cpu().numpy()
        p_np = pred1.cpu().numpy()
        for t, p in zip(y_np, p_np):
            class_tot[t] += 1
            if t == p:
                class_tp[t] += 1

        total += yb.numel()

    top1 = top1_correct / max(total, 1)
    top3 = top3_correct / max(total, 1)

    class_recall = np.divide(
        class_tp, np.maximum(class_tot, 1), out=np.zeros_like(class_tp, dtype=float), where=class_tot > 0
    )
    macro_recall = class_recall[class_tot > 0].mean() if np.any(class_tot > 0) else 0.0

    return {
        "top1": top1,
        "top3": top3,
        "macro_recall": macro_recall,
        "class_recall": class_recall,
        "class_support": class_tot,
    }


def train(args):
    ds = build_il_dataset(args.data_path, include_terminal_diagnose=True)
    (X_train, y_train), (X_test, y_test) = patient_level_split(ds, test_ratio=args.test_ratio, seed=args.seed)

    print(f"Total pairs: {len(ds.y)}")
    print(f"Train pairs: {len(y_train)} | Test pairs: {len(y_test)}")
    print(f"State dim: {ds.X.shape[1]} | Num actions: {len(ACTION_MODALITIES)}")

    train_loader = DataLoader(PairDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(PairDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyNet(in_dim=ds.X.shape[1], n_actions=len(ACTION_MODALITIES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    class_w = make_class_weights(y_train, len(ACTION_MODALITIES))
    class_w_t = torch.tensor(class_w, dtype=torch.float32, device=device)

    if args.loss == "focal":
        criterion = FocalLoss(alpha=class_w_t, gamma=args.gamma)
        print("Using loss: FocalLoss (with class weights)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_w_t)
        print("Using loss: Weighted CrossEntropy")

    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * yb.size(0)

        train_loss = running_loss / max(len(y_train), 1)
        metrics = evaluate(model, test_loader, device, n_classes=len(ACTION_MODALITIES))

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"test_top1={metrics['top1']:.4f} | "
            f"test_top3={metrics['top3']:.4f} | "
            f"test_macro_recall={metrics['macro_recall']:.4f}"
        )

        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "action_names": ACTION_MODALITIES,
                    "state_dim": ds.X.shape[1],
                },
                args.ckpt_path,
            )

    print(f"Best top1: {best_top1:.4f}")
    print(f"Saved best checkpoint to: {args.ckpt_path}")

    # Print per-class recall at end (helpful for imbalance)
    final_metrics = evaluate(model, test_loader, device, n_classes=len(ACTION_MODALITIES))
    print("\nPer-class recall:")
    for i, (rec, sup) in enumerate(zip(final_metrics["class_recall"], final_metrics["class_support"])):
        if sup > 0:
            print(f"{ID2ACTION[i]:24s} recall={rec:.3f} support={int(sup)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/state_trajectories_denoised.json")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, choices=["ce", "focal"], default="ce")
    parser.add_argument("--gamma", type=float, default=2.0)  # focal gamma
    parser.add_argument("--ckpt_path", type=str, default="bc_policy.pt")
    args = parser.parse_args()

    t0 = time.time()
    train(args)
    print(f"elapsed: {time.time()-t0:.2f}s")