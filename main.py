import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from builder import (
    ACTION_MODALITIES,
    ID2ACTION,
    build_il_dataset,
    build_il_dataset_mixed,
    patient_level_split,
    make_class_weights,
)
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from tabpfn import TabPFNClassifier
except Exception:
    TabPFNClassifier = None

# Needed only when number of classes > 10
try:
    from tabpfn_extensions.many_class import ManyClassClassifier
except Exception:
    ManyClassClassifier = None

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

def _align_proba(proba: np.ndarray, classes_present: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    out[:, classes_present.astype(int)] = proba
    return out

def metrics_from_proba(y_true: np.ndarray, proba: np.ndarray, n_classes: int):
    y_pred = np.argmax(proba, axis=1)
    base = metrics_from_preds(y_true, y_pred, n_classes=n_classes)

    k = min(3, n_classes)
    topk_idx = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
    top3 = float((topk_idx == y_true[:, None]).any(axis=1).mean()) if len(y_true) > 0 else 0.0

    base["top3"] = top3
    return base

def metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    total = len(y_true)
    top1 = float((y_true == y_pred).mean()) if total > 0 else 0.0

    class_tp = np.zeros(n_classes, dtype=np.int64)
    class_tot = np.zeros(n_classes, dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        class_tot[t] += 1
        if t == p:
            class_tp[t] += 1

    class_recall = np.divide(
        class_tp, np.maximum(class_tot, 1),
        out=np.zeros_like(class_tp, dtype=float),
        where=class_tot > 0
    )
    macro_recall = class_recall[class_tot > 0].mean() if np.any(class_tot > 0) else 0.0

    return {
        "top1": top1,
        "macro_recall": macro_recall,
        "class_recall": class_recall,
        "class_support": class_tot,
    }


def baseline_majority_predict(y_train: np.ndarray, n_test: int):
    counts = np.bincount(y_train, minlength=len(ACTION_MODALITIES))
    maj = int(np.argmax(counts))
    return np.full(n_test, maj, dtype=np.int64)


def baseline_prior_sample_predict(y_train: np.ndarray, n_test: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    counts = np.bincount(y_train, minlength=len(ACTION_MODALITIES)).astype(np.float64)
    probs = counts / counts.sum()
    return rng.choice(len(probs), size=n_test, p=probs).astype(np.int64)

def baseline_logreg_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int = 42):
    # StandardScaler helps optimization for linear models.
    # with_mean=False keeps it safe if you later switch to sparse features.
    clf = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            multi_class="multinomial",
            solver="saga",
            max_iter=5000,
            class_weight="balanced",   # helps imbalance
            random_state=seed,
            n_jobs=-1,
        ),
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def baseline_xgb_predict_proba(X_train, y_train, X_test, seed=42):
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed. Run: pip install xgboost")

    # Remap sparse/global class IDs -> contiguous local IDs [0..K-1]
    classes_global = np.unique(y_train)
    g2l = {g: i for i, g in enumerate(classes_global)}
    y_train_local = np.array([g2l[g] for g in y_train], dtype=np.int64)

    clf = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        tree_method="hist",
    )
    clf.fit(X_train, y_train_local)
    proba_local = clf.predict_proba(X_test)  # columns are local class IDs

    # Map local columns back to your global action IDs
    return _align_proba(proba_local, classes_global, n_classes=len(ACTION_MODALITIES))


def baseline_tabpfn_predict_proba(X_train, y_train, X_test, seed=42):
    if TabPFNClassifier is None:
        raise RuntimeError("tabpfn is not installed. Run: pip install tabpfn")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = TabPFNClassifier(device=device, random_state=seed)

    classes_global = np.unique(y_train)
    g2l = {g: i for i, g in enumerate(classes_global)}
    y_train_local = np.array([g2l[g] for g in y_train], dtype=np.int64)

    if len(classes_global) > 10:
        if ManyClassClassifier is None:
            raise RuntimeError("Install: pip install tabpfn-extensions")
        clf = ManyClassClassifier(
            estimator=base,
            alphabet_size=10,  # key fix
        )
    else:
        clf = base

    clf.fit(X_train, y_train_local)
    proba = clf.predict_proba(X_test)
    classes_present = getattr(clf, "classes_", np.arange(proba.shape[1]))
    proba = _align_proba(proba, np.asarray(classes_present), n_classes=len(ACTION_MODALITIES))
    return proba


def print_per_class_compare(name_a, rec_a, name_b, rec_b, support, max_rows=30):
    print(f"\nPer-class recall comparison: {name_a} vs {name_b}")
    print(f"{'action':24s} {'support':>7s} {name_a:>10s} {name_b:>10s} {'delta':>10s}")
    shown = 0
    for i, sup in enumerate(support):
        if sup <= 0:
            continue
        delta = rec_a[i] - rec_b[i]
        print(f"{ID2ACTION[i]:24s} {int(sup):7d} {rec_a[i]:10.3f} {rec_b[i]:10.3f} {delta:10.3f}")
        shown += 1
        if shown >= max_rows:
            break


from dataclasses import dataclass
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -------------------------
# Data bundle
# -------------------------
@dataclass
class DataBundle:
    ds: object
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_loader: DataLoader
    test_loader: DataLoader
    device: torch.device


def prepare_data(args) -> DataBundle:
    if args.use_mixed:
        specs = [
            ("chole", args.chole_path, "DIAGNOSE_CHOLECYSTITIS"),
            ("pan", args.pan_path, "DIAGNOSE_PANCREATITIS"),
            ("diver", args.diver_path, "DIAGNOSE_DIVERTICULITIS"),
        ]
        ds = build_il_dataset_mixed(specs, include_terminal_diagnose=True)
    else:
        # single-dataset fallback
        ds = build_il_dataset(
            args.data_path,
            include_terminal_diagnose=True,
            terminal_action="DIAGNOSE_CHOLECYSTITIS",
            dataset_tag="single",
        )

    (X_train, y_train), (X_test, y_test) = patient_level_split(
        ds, test_ratio=args.test_ratio, seed=args.seed
    )

    train_loader = DataLoader(PairDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(PairDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Total pairs: {len(ds.y)}")
    print(f"Train pairs: {len(y_train)} | Test pairs: {len(y_test)}")
    print(f"State dim: {ds.X.shape[1]} | Num actions: {len(ACTION_MODALITIES)}")

    return DataBundle(ds, X_train, y_train, X_test, y_test, train_loader, test_loader, device)


def train(args, b: DataBundle):
    """Pure MLP training + checkpoint saving."""
    model = PolicyNet(in_dim=b.ds.X.shape[1], n_actions=len(ACTION_MODALITIES)).to(b.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    class_w = make_class_weights(b.y_train, len(ACTION_MODALITIES))
    class_w_t = torch.tensor(class_w, dtype=torch.float32, device=b.device)

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

        for xb, yb in b.train_loader:
            xb, yb = xb.to(b.device), yb.to(b.device)
            logits = model(xb)
            loss = criterion(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * yb.size(0)

        train_loss = running_loss / max(len(b.y_train), 1)
        metrics = evaluate(model, b.test_loader, b.device, n_classes=len(ACTION_MODALITIES))

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
                    "state_dim": b.ds.X.shape[1],
                },
                args.ckpt_path,
            )

    print(f"Best top1: {best_top1:.4f}")
    print(f"Saved best checkpoint to: {args.ckpt_path}")


def eval_mlp_best(args, b: DataBundle):
    model = PolicyNet(in_dim=b.ds.X.shape[1], n_actions=len(ACTION_MODALITIES)).to(b.device)
    ckpt = torch.load(args.ckpt_path, map_location=b.device)
    model.load_state_dict(ckpt["model_state_dict"])
    m = evaluate(model, b.test_loader, b.device, n_classes=len(ACTION_MODALITIES))

    print("\n=== MLP (best checkpoint) ===")
    print(f"top1={m['top1']:.4f} top3={m['top3']:.4f} macro_recall={m['macro_recall']:.4f}")
    return m



def compare(args, b: DataBundle, mlp_metrics: dict):
    """Compare MLP vs 3 baselines: majority, prior-sampling, logistic-regression."""
    # Baseline 1: majority
    y_pred_major = baseline_majority_predict(b.y_train, len(b.y_test))
    major = metrics_from_preds(b.y_test, y_pred_major, n_classes=len(ACTION_MODALITIES))
    print("\n=== Baseline: Majority ===")
    print(f"top1={major['top1']:.4f} macro_recall={major['macro_recall']:.4f}")

    # Baseline 2: prior sampling
    y_pred_prior = baseline_prior_sample_predict(b.y_train, len(b.y_test), seed=args.seed)
    prior = metrics_from_preds(b.y_test, y_pred_prior, n_classes=len(ACTION_MODALITIES))
    print("\n=== Baseline: Class-prior sampling ===")
    print(f"top1={prior['top1']:.4f} macro_recall={prior['macro_recall']:.4f}")

    # Baseline 3: logistic regression
    y_pred_logreg = baseline_logreg_predict(b.X_train, b.y_train, b.X_test, seed=args.seed)
    logreg = metrics_from_preds(b.y_test, y_pred_logreg, n_classes=len(ACTION_MODALITIES))  
    
    # after logreg:
    print("\n=== Baseline: Logistic Regression ===")
    print(f"top1={logreg['top1']:.4f} macro_recall={logreg['macro_recall']:.4f}")

    if args.run_xgb:
        xgb_proba = baseline_xgb_predict_proba(b.X_train, b.y_train, b.X_test, seed=args.seed)
        xgb = metrics_from_proba(b.y_test, xgb_proba, n_classes=len(ACTION_MODALITIES))
        print("\n=== Baseline: XGBoost ===")
        print(f"top1={xgb['top1']:.4f} top3={xgb['top3']:.4f} macro_recall={xgb['macro_recall']:.4f}")

    if args.run_tabpfn:
        tabpfn_proba = baseline_tabpfn_predict_proba(b.X_train, b.y_train, b.X_test, seed=args.seed)
        tabpfn = metrics_from_proba(b.y_test, tabpfn_proba, n_classes=len(ACTION_MODALITIES))
        print("\n=== Baseline: TabPFN ===")
        print(f"top1={tabpfn['top1']:.4f} top3={tabpfn['top3']:.4f} macro_recall={tabpfn['macro_recall']:.4f}")

def compare_per_class(args, b: DataBundle, mlp_metrics: dict):
    n_cls = len(ACTION_MODALITIES)

    # 1) Compute baseline metrics
    y_pred_logreg = baseline_logreg_predict(b.X_train, b.y_train, b.X_test, seed=args.seed)
    logreg = metrics_from_preds(b.y_test, y_pred_logreg, n_classes=n_cls)

    xgb = None
    if args.run_xgb:
        xgb_proba = baseline_xgb_predict_proba(b.X_train, b.y_train, b.X_test, seed=args.seed)
        xgb = metrics_from_proba(b.y_test, xgb_proba, n_classes=n_cls)

    tabpfn = None
    if args.run_tabpfn:
        tabpfn_proba = baseline_tabpfn_predict_proba(b.X_train, b.y_train, b.X_test, seed=args.seed)
        tabpfn = metrics_from_proba(b.y_test, tabpfn_proba, n_classes=n_cls)

    # 2) Per-class recall comparisons (same support from test set)
    print_per_class_compare(
        "MLP", mlp_metrics["class_recall"],
        "LogReg", logreg["class_recall"],
        mlp_metrics["class_support"],
        max_rows=999,
    )

    if xgb is not None:
        print_per_class_compare(
            "MLP", mlp_metrics["class_recall"],
            "XGBoost", xgb["class_recall"],
            mlp_metrics["class_support"],
            max_rows=999,
        )

    if tabpfn is not None:
        print_per_class_compare(
            "MLP", mlp_metrics["class_recall"],
            "TabPFN", tabpfn["class_recall"],
            mlp_metrics["class_support"],
            max_rows=999,
        )

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
    parser.add_argument("--chole_path", type=str, default="data_chole/state_trajectories_denoised_chole.json")
    parser.add_argument("--pan_path", type=str, default="data_pan/state_trajectories_denoised_pan.json")
    parser.add_argument("--diver_path", type=str, default="data_diver/state_trajectories_denoised_diver.json")
    parser.add_argument("--use_mixed", action="store_true")
    parser.add_argument("--run_xgb", action="store_true")
    parser.add_argument("--run_tabpfn", action="store_true")
    args = parser.parse_args()

    b = prepare_data(args)
    t0 = time.time()
    train(args, b)
    mlp_metrics = eval_mlp_best(args, b)
    compare(args, b, mlp_metrics)
    compare_per_class(args, b, mlp_metrics)
    print(f"elapsed: {time.time()-t0:.2f}s")

#python main.py --use_mixed --run_xgb --run_tabpfn