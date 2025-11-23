import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (classification_report, precision_recall_fscore_support,
                             confusion_matrix, matthews_corrcoef, roc_auc_score,
                             average_precision_score, balanced_accuracy_score,
                             roc_curve, auc, precision_recall_curve)
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import json
import time
import random
import os
import sys

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
train_csv = root / "train_scaled_binary.csv"
val_csv = root / "val_scaled_binary.csv"
test_csv = root / "test_scaled_binary.csv"


# =========================================================
# 3. Data Reading & Partitioning
# =========================================================
def load_xy(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['delivery_risk']).values.astype('float32')
    y = df['delivery_risk'].values.astype('int64')
    return torch.tensor(X), torch.tensor(y)


X_train, y_train = load_xy(train_csv)
X_val, y_val = load_xy(val_csv)
X_test, y_test = load_xy(test_csv)
NUM_FEATURES = X_train.shape[1]

batch_size = 512
train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=batch_size, shuffle=True, drop_last=True,
                          generator=g)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size,generator=g)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size,generator=g)


# =========================================================
# 4. Component Ablation Experimental Model
# =========================================================

class ComponentAblationModel(nn.Module):
    def __init__(self, in_dim=43, num_classes=2, dim=128, heads=4, depth=3, ablation_type="full"):
        super().__init__()
        self.ablation_type = ablation_type

        if ablation_type == "full":
            # CNN + MLP + Transformer + Attention Pooling
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

        elif ablation_type == "no_cnn":
            #MLP + Transformer
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

        elif ablation_type == "no_mlp":
            # CNN + Transformer
            self.local_path = nn.Sequential(
                nn.Conv1d(1, 64, 3, padding=1, groups=1),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, dim, 3, padding=1, groups=4),
                nn.BatchNorm1d(dim), nn.ReLU()
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

        elif ablation_type == "no_cnn_mlp":
            # Only Transformer
            self.input_proj = nn.Linear(in_dim, dim)
            self.pos_embed = nn.Parameter(torch.randn(1, in_dim, dim))
            encoder = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                dropout=0.9, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder, num_layers=depth)
            self.attn_pool = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)

            self.head = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.ReLU(),
                nn.Dropout(0.9), nn.Linear(dim // 2, num_classes))

    def forward(self, x):
        if self.ablation_type == "full":
            B, L = x.shape
            local = self.local_path(x.unsqueeze(1)).transpose(1, 2)
            global_ = self.global_path(x).unsqueeze(1).expand(-1, L, -1)
            z = local + global_ + self.pos_embed
            z = self.transformer(z)
            query = z.mean(dim=1, keepdim=True)
            out, _ = self.attn_pool(query, z, z)
            return self.head(out.squeeze(1))

        elif self.ablation_type == "no_cnn":
            B, L = x.shape
            global_ = self.global_path(x).unsqueeze(1).expand(-1, L, -1)
            z = global_ + self.pos_embed
            z = self.transformer(z)
            query = z.mean(dim=1, keepdim=True)
            out, _ = self.attn_pool(query, z, z)
            return self.head(out.squeeze(1))

        elif self.ablation_type == "no_mlp":
            B, L = x.shape
            local = self.local_path(x.unsqueeze(1)).transpose(1, 2)
            z = local + self.pos_embed
            z = self.transformer(z)
            query = z.mean(dim=1, keepdim=True)
            out, _ = self.attn_pool(query, z, z)
            return self.head(out.squeeze(1))

        elif self.ablation_type == "no_cnn_mlp":
            B, L = x.shape
            projected = self.input_proj(x).unsqueeze(1).expand(-1, L, -1)
            z = projected + self.pos_embed
            z = self.transformer(z)
            query = z.mean(dim=1, keepdim=True)
            out, _ = self.attn_pool(query, z, z)
            return self.head(out.squeeze(1))


# =========================================================
# 5. Assessment Tool
# =========================================================
def plot_confusion_matrix(results, timestamp):
    """Confusion Matrix for the Complete Model"""
    full_result = next((r for r in results if r['ablation_type'] == 'full'), None)
    if not full_result:
        print("No complete model results were found; therefore, the confusion matrix cannot be plotted.")
        return

    cm = np.array(full_result['confusion_matrix'])
    class_names = ['Not Late', 'Late']

    plt.figure(figsize=(6, 5))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Complete Model Confusion Matrix', fontsize=14)
    plt.xlabel('Prediction Label')
    plt.ylabel('Authenticity Label')
    plt.tight_layout()
    cm_filename = f'confusion_matrix_full_model_{timestamp}.png'
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"The complete model confusion matrix chart has been saved as: {cm_filename}")


def detailed_metrics(y_true, y_pred, class_names=None):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    accuracy = np.mean(y_true == y_pred)
    cm = confusion_matrix(y_true, y_pred)
    fpr = []  # False Positive Rate
    fnr = []  # False Negative Rate
    for i in range(len(cm)):
        tn = np.sum(cm) - np.sum(cm[i]) - np.sum(cm[:, i]) + cm[i, i]  # True Negative
        fp = np.sum(cm[:, i]) - cm[i, i]  # False Positive
        fn = np.sum(cm[i]) - cm[i, i]  # False Negative
        tp = cm[i, i]  # True Positive

        fpr_i = fp / (fp + tn) if (fp + tn) > 0 else 0  # FPR = FP / (FP + TN)
        fnr_i = fn / (fn + tp) if (fn + tp) > 0 else 0  # FNR = FN / (FN + TP)

        fpr.append(fpr_i)
        fnr.append(fnr_i)
    specificity = [(np.sum(cm) - np.sum(cm[i]) - np.sum(cm[:, i]) + cm[i, i]) /
                   (np.sum(cm) - np.sum(cm[i]) - np.sum(cm[:, i]) + cm[i, i] + (np.sum(cm[:, i]) - cm[i, i]))
                   if (np.sum(cm) - np.sum(cm[i]) - np.sum(cm[:, i]) + cm[i, i] + (np.sum(cm[:, i]) - cm[i, i])) > 0
                   else 0 for i in range(len(cm))]
    metrics = {
        'accuracy': accuracy,
        'macro_precision': np.mean(precision), 'macro_recall': np.mean(recall),
        'macro_f1': np.mean(f1), 'macro_specificity': np.mean(specificity),
        'macro_fpr': np.mean(fpr), 'macro_fnr': np.mean(fnr),
        'weighted_precision': np.average(precision, weights=support),
        'weighted_recall': np.average(recall, weights=support),
        'weighted_f1': np.average(f1, weights=support),
        'weighted_specificity': np.average(specificity, weights=support),
        'weighted_fpr': np.average(fpr, weights=support), 'weighted_fnr': np.average(fnr, weights=support),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'confusion_matrix': cm.tolist()
    }
    for i, name in enumerate(class_names):
        metrics[f'{name}_precision'], metrics[f'{name}_recall'] = precision[i], recall[i]
        metrics[f'{name}_f1'], metrics[f'{name}_specificity'] = f1[i], specificity[i]
        metrics[f'{name}_fpr'], metrics[f'{name}_fnr'] = fpr[i], fnr[i]
        metrics[f'{name}_support'] = support[i]
    return metrics


# =========================================================
# 6. Training-related tools
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.65, gamma=3):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


def run_epoch(loader, train=False, model=None, optimizer=None, criterion=None, device=None):
    model.train(train)
    total_loss, total_acc, n = 0, 0, 0
    all_preds, all_labels = [], []
    for x, y in tqdm(loader, leave=False, desc="Train" if train else "Eval"):
        x, y = x.to(device), y.to(device)
        logits = model(x)

        if criterion is not None:
            loss = criterion(logits, y)
        else:
            loss = F.cross_entropy(logits, y)
        if train and optimizer is not None:
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
    avg_loss = total_loss / n if n > 0 else 0
    avg_acc = total_acc / n if n > 0 else 0
    return (avg_loss, avg_acc, all_labels, all_preds) if not train else (avg_loss, avg_acc)


# =========================================================
# 7. Component Ablation Experiment Run Function
# =========================================================

def run_single_component_ablation_experiment(ablation_type, experiment_name, train_loader, val_loader, test_loader):
    """Run a Single Component Ablation Experiment"""
    print(f"\n{'=' * 60}")
    print(f"Initiate Component Ablation Experiment: {experiment_name} ({ablation_type})")
    print(f"{'=' * 60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Use of Equipment: {device}")

    model = ComponentAblationModel(
        in_dim=NUM_FEATURES,
        dim=128,
        heads=4,
        depth=3,
        ablation_type=ablation_type
    ).to(device)

    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    criterion = FocalLoss(alpha=0.65, gamma=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=600, factor=0.5, min_lr=1e-8, verbose=False)
    best_val_f1, patience_cnt, patience_limit = 0, 0, 30
    training_start_time = time.time()

    for epoch in range(100):
        train_loss, train_acc = run_epoch(train_loader, train=True, model=model, optimizer=optimizer,
                                          criterion=criterion, device=device)
        val_loss, val_acc, val_y, val_pred = run_epoch(val_loader, train=False, model=model, criterion=criterion,
                                                       device=device)
        val_metrics = detailed_metrics(val_y, val_pred)
        val_f1 = val_metrics['macro_f1']
        scheduler.step(val_f1)

        if epoch % 10 == 0 or val_f1 > best_val_f1:
            print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"best_model_{experiment_name}.pt")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience_limit:
                print("Early stop triggered")
                break

    training_time = time.time() - training_start_time
    print(f"Training completed, time taken: {training_time:.2f}second")
    print("Begin evaluation of the test set...")
    model.load_state_dict(torch.load(f"best_model_{experiment_name}.pt", weights_only=True))

    test_loss, test_acc, test_y, test_pred = run_epoch(test_loader, train=False, model=model, criterion=criterion,
                                                       device=device)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    y_prob = np.concatenate(all_probs, axis=0)

    class_names = ['Not Late', 'Late']
    test_metrics = detailed_metrics(test_y, test_pred, class_names=class_names)
    test_metrics['roc_auc'] = roc_auc_score(test_y, y_prob[:, 1])
    test_metrics['pr_auc'] = average_precision_score(test_y, y_prob[:, 1])

    # Check the predicted distribution
    unique, counts = np.unique(test_pred, return_counts=True)
    print(f"Prediction Distribution: {dict(zip(unique, counts))}")
    print(f"True Distribution: {dict(zip(*np.unique(test_y, return_counts=True)))}")

    correct_predictions = np.sum(np.array(test_y) == np.array(test_pred))
    total_predictions = len(test_y)
    recalculated_accuracy = correct_predictions / total_predictions
    print(f"Recalculated Accuracy: {recalculated_accuracy:.4f}")

    results = {
        'experiment_name': experiment_name,
        'ablation_type': ablation_type,
        'test_accuracy': recalculated_accuracy,
        'test_precision': test_metrics['macro_precision'],
        'test_recall': test_metrics['macro_recall'],
        'test_f1': test_metrics['macro_f1'],
        'test_roc_auc': test_metrics['roc_auc'],
        'test_pr_auc': test_metrics['pr_auc'],
        'test_mcc': test_metrics['mcc'],
        'test_balanced_accuracy': test_metrics['balanced_accuracy'],
        'test_macro_fpr': test_metrics['macro_fpr'],
        'test_macro_fnr': test_metrics['macro_fnr'],
        'test_weighted_fpr': test_metrics['weighted_fpr'],
        'test_weighted_fnr': test_metrics['weighted_fnr'],
        'confusion_matrix': test_metrics['confusion_matrix'],
        'training_time': training_time,
        'prediction_distribution': dict(zip(unique, counts)) if len(unique) > 0 else {}
    }

    print(f"\n{experiment_name} Test results:")
    print(f"Accuracy: {results['test_accuracy']:.4f}")
    print(f"Precision: {results['test_precision']:.4f}")
    print(f"Recall: {results['test_recall']:.4f}")
    print(f"F1-Score: {results['test_f1']:.4f}")
    print(f"ROC-AUC: {results['test_roc_auc']:.4f}")
    print(f"PR-AUC: {results['test_pr_auc']:.4f}")
    print(f"MCC: {results['test_mcc']:.4f}")
    print(f"Balanced Accuracy: {results['test_balanced_accuracy']:.4f}")

    print(f"Macro FPR: {test_metrics['macro_fpr']:.4f}")
    print(f"Macro FNR: {test_metrics['macro_fnr']:.4f}")
    print(f"Weighted FPR: {test_metrics['weighted_fpr']:.4f}")
    print(f"Weighted FNR: {test_metrics['weighted_fnr']:.4f}")

    for i, name in enumerate(class_names):
        print(f"{name} FPR: {test_metrics[f'{name}_fpr']:.4f}")
        print(f"{name} FNR: {test_metrics[f'{name}_fnr']:.4f}")

    print(f"Prediction Distribution: {results['prediction_distribution']}")

    return results


# =========================================================
# 8. Batch Component Ablation Experiment Run
# =========================================================

def run_all_component_ablation_experiments():
    """Run all component ablation experiments"""
    ablation_configs = [
        ("full", "LGFT"),
        ("no_cnn", "Remove cnn"),
        ("no_mlp", "Remove mlp"),
        ("no_cnn_mlp", "Remove cnn+mlp")
    ]

    all_results = []

    print("Initiate batch component ablation experiments...")
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}, Test set size: {len(X_test)}")
    print(f"Feature Dimension: {NUM_FEATURES}")

    for ablation_type, experiment_name in ablation_configs:
        try:
            result = run_single_component_ablation_experiment(ablation_type, experiment_name, train_loader, val_loader,
                                                              test_loader)
            all_results.append(result)
            print(f"Experiment {experiment_name} Completed")
        except Exception as e:
            print(f"Experiment {experiment_name} Failure: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 120}")
    print("Component Ablation Experiment Results Comparison")
    print(f"{'=' * 120}")
    print(
        f"{'Model Name':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'PR-AUC':<10} {'MCC':<10} {'Balanced Acc':<12} {'Weighted FPR':<12} {'Weighted FNR':<12} {'Training Time(s)':<12}")
    print("-" * 160)

    for result in all_results:
        print(f"{result['experiment_name']:<20} "
              f"{result['test_accuracy']:<10.4f} "
              f"{result['test_precision']:<10.4f} "
              f"{result['test_recall']:<10.4f} "
              f"{result['test_f1']:<10.4f} "
              f"{result['test_roc_auc']:<10.4f} "
              f"{result['test_pr_auc']:<10.4f} "
              f"{result['test_mcc']:<10.4f} "
              f"{result['test_balanced_accuracy']:<12.4f} "
              f"{result.get('test_weighted_fpr', 0):<10.4f} "
              f"{result.get('test_weighted_fnr', 0):<10.4f}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f'component_ablation_results_{timestamp}.json'
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results have been saved to: {results_filename}")

    plot_component_ablation_results(all_results, timestamp)
    plot_confusion_matrix(all_results, timestamp)
    plot_component_ablation_results(all_results, timestamp)
    return all_results


def plot_component_ablation_results(results, timestamp):
    """Plot a bar chart comparing component ablation experiment results (Accuracy, Precision, Recall, F1)"""
    if not results:
        print("No results to display")
        return

    experiment_names = [r['experiment_name'] for r in results]
    metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    # Extract the metric values for each model
    data = {metric: [r[metric] for r in results] for metric in metrics}

    x = np.arange(len(experiment_names))
    width = 0.2

    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = plt.bar(x + i * width, data[metric], width, label=metric_names[i], color=color)
        # Add values to the top of the columns
        for bar, value in zip(bars, data[metric]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Model')
    plt.ylabel('Indicator value')
    plt.title('Comparison of Ablation Experiment Results')
    plt.xticks(x + width * 1.5, experiment_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_filename = f'model_comparison_metrics_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Model Metrics Comparison Chart has been saved as: {plot_filename}")


# =========================================================
# 9. Model Complexity Analysis
# =========================================================

def analyze_component_model_complexity():
    """Analyze the complexity of each component model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models_info = {
        "FULL": ComponentAblationModel(NUM_FEATURES, dim=128, heads=4, depth=3, ablation_type="full"),
        "W/O_CNN": ComponentAblationModel(NUM_FEATURES, dim=128, heads=4, depth=3, ablation_type="no_cnn"),
        "W/O_MLP": ComponentAblationModel(NUM_FEATURES, dim=128, heads=4, depth=3, ablation_type="no_mlp"),
        "W/O_CNN+MLP": ComponentAblationModel(NUM_FEATURES, dim=128, heads=4, depth=3, ablation_type="no_cnn_mlp")
    }

    print(f"\n{'=' * 80}")
    print("Component Model Complexity Analysis")
    print(f"{'=' * 80}")
    print(f"{'Model Name':<20} {'Number of parameters':<15} {'Estimated size(MB)':<15} {'Relatively Complete Model':<15}")
    print("-" * 80)

    full_model_params = None
    for name, model in models_info.items():
        model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * 4 / (1024 * 1024)

        if full_model_params is None:
            full_model_params = param_count
            relative_size = "Benchmark(100%)"
        else:
            relative_size = f"{(param_count / full_model_params) * 100:.1f}%"

        print(f"{name:<20} {param_count:<15,} {model_size_mb:<15.2f} {relative_size:<15}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =========================================================
# 10. Component Contribution Analysis
# =========================================================

def analyze_component_contributions(results):
    """Analyze the contribution of each component"""
    if not results:
        print("There are no results to analyze.")
        return

    print(f"\n{'=' * 60}")
    print("Component Contribution Analysis")
    print(f"{'=' * 60}")

    full_model_result = next((r for r in results if r['ablation_type'] == 'full'), None)
    if not full_model_result:
        print("No complete model results found")
        return

    print(f"Complete Model Performance:")
    print(f"  Accuracy: {full_model_result['test_accuracy']:.4f}")
    print(f"  F1-Score: {full_model_result['test_f1']:.4f}")
    print(f"  ROC-AUC: {full_model_result['test_roc_auc']:.4f}")
    print(f"  MCC: {full_model_result['test_mcc']:.4f}")

    print(f"\nPerformance degradation after component removal:")
    print(f"{'component':<15} {'Accuracy decline':<12} {'F1 decline':<10} {'ROC-AUC decline':<12} {'MCC decline':<10}")
    print("-" * 60)

    for result in results:
        if result['ablation_type'] != 'full':
            acc_drop = full_model_result['test_accuracy'] - result['test_accuracy']
            f1_drop = full_model_result['test_f1'] - result['test_f1']
            roc_drop = full_model_result['test_roc_auc'] - result['test_roc_auc']
            mcc_drop = full_model_result['test_mcc'] - result['test_mcc']

            print(f"{result['experiment_name']:<15} "
                  f"{acc_drop:<12.4f} "
                  f"{f1_drop:<10.4f} "
                  f"{roc_drop:<12.4f} "
                  f"{mcc_drop:<10.4f}")


# =========================================================
# 11. Main Function
# =========================================================

def main():
    print("Supply Chain Delay Detection Component Dissolution Test")
    print("=" * 80)
    print("Experimental Setup:")
    print(f"  - Data set path: {root}")
    print(f"  - Feature Dimension: {NUM_FEATURES}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Model Parameters: dim=128, heads=4, depth=3")
    print("=" * 80)

    # Analysis of Model Complexity
    analyze_component_model_complexity()

    # Running Component Ablation Experiment
    print("\nInitiate the component ablation experiment....")
    results = run_all_component_ablation_experiments()

    # Generate Summary Report
    if results:
        # Component Contribution Analysis
        analyze_component_contributions(results)

        print(f"\n{'=' * 60}")
        print("Experiment Summary")
        print(f"{'=' * 60}")

        best_result = max(results, key=lambda x: x['test_f1'])
        print(f"   Best Model: {best_result['experiment_name']}")
        print(f"   F1-Score: {best_result['test_f1']:.4f}")
        print(f"   ROC-AUC: {best_result['test_roc_auc']:.4f}")
        print(f"   MCC: {best_result['test_mcc']:.4f}")

        print(f"\n Performance Comparison:")
        full_model_result = next((r for r in results if r['ablation_type'] == 'FULL'), None)
        if full_model_result:
            no_cnn_result = next((r for r in results if r['ablation_type'] == 'no_cnn'), None)
            no_mlp_result = next((r for r in results if r['ablation_type'] == 'no_mlp'), None)
            no_cnn_mlp_result = next((r for r in results if r['ablation_type'] == 'no_cnn_mlp'), None)

            if no_cnn_result:
                f1_diff = full_model_result['test_f1'] - no_cnn_result['test_f1']
                print(f"   F1 score decreased after removing CNN: {f1_diff:.4f}")

            if no_mlp_result:
                f1_diff = full_model_result['test_f1'] - no_mlp_result['test_f1']
                print(f"   F1 score decreased after removing MLP: {f1_diff:.4f}")

            if no_cnn_mlp_result:
                f1_diff = full_model_result['test_f1'] - no_cnn_mlp_result['test_f1']
                print(f"   F1 score decreased after removing CNN+MLP: {f1_diff:.4f}")

    print(f"\n Component ablation experiment completed")



if __name__ == "__main__":

    if len(sys.argv) > 1:
        experiment_type = sys.argv[1]
        if experiment_type in ["full", "no_cnn", "no_mlp", "no_cnn_mlp"]:
            result = run_single_component_ablation_experiment(
                experiment_type,
                {"full": "FULL",
                 "no_cnn": "W/O CNN",
                 "no_mlp": "W/O MLP",
                 "no_cnn_mlp": "W/O CNN+MLP"}[experiment_type],
                train_loader, val_loader, test_loader
            )
            print(f"Individual experimental results: {result}")
        else:
            print("python component_ablation_experiment.py [full|no_cnn|no_mlp|no_cnn_mlp]")
    else:
        main()