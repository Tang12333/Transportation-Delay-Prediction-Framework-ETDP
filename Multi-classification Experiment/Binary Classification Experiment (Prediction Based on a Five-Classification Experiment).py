import pandas as pd
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (classification_report, precision_recall_fscore_support,
                             confusion_matrix, matthews_corrcoef, roc_auc_score,
                             average_precision_score, balanced_accuracy_score,
                             roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import random
import os
import time
import shap

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

g = torch.Generator()
g.manual_seed(RANDOM_SEED)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 1. Data set path
# =========================================================
root = Path(r"...")
train_csv = root / "SCRM_train_mix.csv"
val_csv   = root / "SCRM_val_mix.csv"
test_csv  = root / "SCRM_test_mix.csv"

# =========================================================
# 2. Read & Partition
# =========================================================
def load_xy(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['SCMstability_category']).values.astype('float32')
    y = df['SCMstability_category'].values.astype('int64')
    return torch.tensor(X), torch.tensor(y)

X_train, y_train = load_xy(train_csv)
X_val, y_val     = load_xy(val_csv)
X_test, y_test   = load_xy(test_csv)
NUM_FEATURES = X_train.shape[1]

batch_size = 512
train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=batch_size, shuffle=True, drop_last=True,
                          generator=g)
val_loader   = DataLoader(TensorDataset(X_val, y_val),   batch_size=batch_size,
                         generator=g)
test_loader  = DataLoader(TensorDataset(X_test, y_test),  batch_size=batch_size,
                         generator=g)

# =========================================================
# 3. Model Definition
# =========================================================
class ETDP(nn.Module):
    def __init__(self, in_dim=30, num_classes=5, dim=128, heads=4, depth=3):
        super().__init__()

        self.local_path = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1, groups=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, dim, 3, padding=1, groups=4),
            nn.BatchNorm1d(dim), nn.ReLU()
        )

        self.global_path = nn.Sequential(
            nn.Linear(in_dim, dim * 2), nn.ReLU(),
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim)
        )

        self.pos_embed = nn.Parameter(torch.randn(1, in_dim, dim))

        encoder = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=depth)
        self.attn_pool = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(dim // 2, num_classes))

    def forward(self, x):
        B, L = x.shape
        local  = self.local_path(x.unsqueeze(1)).transpose(1, 2)
        global_= self.global_path(x).unsqueeze(1).expand(-1, L, -1)
        z = local + global_ + self.pos_embed
        z = self.transformer(z)
        query = z.mean(dim=1, keepdim=True)
        out, _ = self.attn_pool(query, z, z)
        return self.head(out.squeeze(1))

# =========================================================
# 4. Assessment Tool
# =========================================================
def detailed_metrics(y_true, y_pred, class_names=None):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    accuracy = np.mean(y_true == y_pred)
    cm = confusion_matrix(y_true, y_pred)

    specificity = []
    fpr_list = []
    fnr_list = []
    for i in range(len(cm)):
        tn = np.sum(cm) - np.sum(cm[i]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        fn = np.sum(cm[i]) - cm[i, i]
        tp = cm[i, i]

        specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(specificity_i)

        fpr_i = fp / (fp + tn) if (fp + tn) > 0 else 0
        fpr_list.append(fpr_i)

        fnr_i = fn / (fn + tp) if (fn + tp) > 0 else 0
        fnr_list.append(fnr_i)

    metrics = {
        'accuracy': accuracy,
        'macro_precision': np.mean(precision),
        'macro_recall': np.mean(recall),
        'macro_f1': np.mean(f1),
        'macro_specificity': np.mean(specificity),
        'macro_fpr': np.mean(fpr_list),
        'macro_fnr': np.mean(fnr_list),
        'weighted_precision': np.average(precision, weights=support),
        'weighted_recall': np.average(recall, weights=support),
        'weighted_f1': np.average(f1, weights=support),
        'weighted_specificity': np.average(specificity, weights=support),
        'weighted_fpr': np.average(fpr_list, weights=support),
        'weighted_fnr': np.average(fnr_list, weights=support),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'roc_auc': None,
        'pr_auc': None,
        'confusion_matrix': cm.tolist()
    }

    for i, name in enumerate(class_names):
        metrics[f'{name}_precision'], metrics[f'{name}_recall'] = precision[i], recall[i]
        metrics[f'{name}_f1'], metrics[f'{name}_specificity'] = f1[i], specificity[i]
        metrics[f'{name}_fpr'], metrics[f'{name}_fnr'] = fpr_list[i], fnr_list[i]
        metrics[f'{name}_support'] = support[i]

    return metrics

# =========================================================
# 5. Drawing Tools
# =========================================================
def plot_roc_curves(y_true, y_prob, class_names):
    plt.figure(figsize=(8, 6))

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    all_fpr = []
    all_tpr = []

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC={roc_auc:.3f})')

        all_fpr.append(fpr)
        all_tpr.append(tpr)

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False positive rate', fontsize=12)
    plt.ylabel('True Rate', fontsize=12)
    plt.title('Multi-class ROC Curve (OvR)', fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve_multiclass.png', dpi=300)
    plt.show()


def plot_pr_curves(y_true, y_prob, class_names):
    plt.figure(figsize=(8, 6))

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    for i in range(n_classes):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        plt.plot(rec, prec, lw=2, label=f'{class_names[i]} (AP={ap:.3f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precious', fontsize=12)
    plt.title('Multi-class PR Curve (OvR)', fontsize=14)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('pr_curve_multiclass.png', dpi=300)
    plt.show()

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

# =========================================================
# 6. Loss & Train
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.65, gamma=3):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ETDP(in_dim=NUM_FEATURES, dim=128, heads=4, depth=3).to(device)
criterion = FocalLoss(alpha=0.65, gamma=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=600, factor=0.5, min_lr=1e-8, verbose=True)

def run_epoch(loader, train=False, model=None, optimizer=None, criterion=None):
    if model is None:      model = globals()['model']
    if optimizer is None:  optimizer = globals()['optimizer']
    if criterion is None:  criterion = globals()['criterion']
    model.train(train)
    total_loss, total_acc, n = 0, 0, 0
    all_preds, all_labels = [], []
    for x, y in tqdm(loader, leave=False, desc="Train" if train else "Eval"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(1)
        total_acc += (preds == y).sum().item()
        n += y.size(0)
        if not train:
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / n
    avg_acc = total_acc / n
    return (avg_loss, avg_acc, all_labels, all_preds) if not train else (avg_loss, avg_acc)

train_start_time = time.time()
best_val_f1, patience_cnt, patience_limit = 0, 0, 30
for epoch in range(100):
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc, val_y, val_pred = run_epoch(val_loader, train=False)
    val_metrics = detailed_metrics(val_y, val_pred)
    val_f1 = val_metrics['macro_f1']
    scheduler.step(val_f1)
    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_optimized.pt")
        patience_cnt = 0
        print(f" New best F1 score: {best_val_f1:.4f}")
    else:
        patience_cnt += 1
        if patience_cnt >= patience_limit:
            print("Early stop triggered")
            break

train_elapsed = time.time() - train_start_time
# =========================================================
# 7. Testing + Drawing
# =========================================================
model.load_state_dict(torch.load("best_optimized.pt"))
test_loss, test_acc, test_y, test_pred = run_epoch(test_loader, train=False)

model.eval()
all_probs = []
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(device)
        all_probs.append(torch.softmax(model(x), dim=1).cpu().numpy())
y_prob = np.concatenate(all_probs, axis=0)

unique_classes = np.unique(test_y)
num_classes_found = len(unique_classes)

class_names = [
    'class 0',
    'class 1',
    'class 2',
    'class 3',
    'class 4'
][:num_classes_found]

test_metrics = detailed_metrics(test_y, test_pred, class_names=class_names)
report = classification_report(
    test_y,
    test_pred,
    labels=unique_classes,
    target_names=class_names,
    digits=4
)
print(report)

cm = confusion_matrix(test_y, test_pred, labels=unique_classes)
N = cm.sum()

print("\n[Per-class Accuracy (One-vs-Rest)]")
hdr = f"{'class':<12}{'accuracy':>12}{'support':>12}"
print(hdr)
print("-" * len(hdr))

for i, cname in enumerate(class_names):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = N - TP - FP - FN
    acc_i = (TP + TN) / N if N > 0 else 0.0
    support_i = int(cm[i, :].sum())
    print(f"{cname:<12}{acc_i:>12.4f}{support_i:>12d}")


print(f"\nOverall Performance Metrics:")
print(f"Macro Average Precision: {test_metrics['macro_precision']:.4f}")
print(f"Macro Average Recall: {test_metrics['macro_recall']:.4f}")
print(f"Macro Average F1-Score: {test_metrics['macro_f1']:.4f}")
print(f"Weighted Fpr: {test_metrics['weighted_fpr']:.4f}")
print(f"Weighted Fnr: {test_metrics['weighted_fnr']:.4f}")

print(f"accuracy: {test_acc:.4f} | Training duration: {train_elapsed:.2f}s")
test_metrics['roc_auc'] = roc_auc_score(
    test_y,
    y_prob,
    multi_class='ovr',
    average='weighted',
    labels=unique_classes
)

y_true_binarized = label_binarize(test_y, classes=unique_classes)

test_metrics['pr_auc'] = average_precision_score(
    y_true=y_true_binarized,
    y_score=y_prob,
    average='weighted'
)


print(f"\n{'=' * 50}")
print("FINAL TEST RESULTS")
print(f"{'=' * 50}")
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
print(f"MCC: {test_metrics['mcc']:.4f}")
print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
print(f"PR-AUC: {test_metrics['pr_auc']:.4f}")

cm = np.array(test_metrics['confusion_matrix'])
plot_confusion_matrix(cm, class_names)
plot_roc_curves(test_y, y_prob, class_names)
plot_pr_curves(test_y, y_prob, class_names)

