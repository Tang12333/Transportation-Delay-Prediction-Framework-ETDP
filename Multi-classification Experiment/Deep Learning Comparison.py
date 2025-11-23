import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (classification_report, precision_recall_fscore_support,
                             confusion_matrix, matthews_corrcoef, roc_auc_score,
                             average_precision_score, balanced_accuracy_score)
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import json
import time
import random
import os

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

# Dataset paths
root = Path(r"...")
train_csv = root / "SCRM_train_mix.csv"
val_csv = root / "SCRM_val_mix.csv"
test_csv = root / "SCRM_test_mix.csv"

# Data loading function
def load_xy(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['SCMstability_category']).values.astype('float32')
    y = df['SCMstability_category'].values.astype('int64')
    return torch.tensor(X), torch.tensor(y)

X_train, y_train = load_xy(train_csv)
X_val, y_val = load_xy(val_csv)
X_test, y_test = load_xy(test_csv)
NUM_FEATURES = X_train.shape[1]

batch_size = 512
train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=batch_size, shuffle=True, drop_last=True,
                          generator=g)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size,
                        generator=g)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size,
                         generator=g)

# Traditional deep learning model definitions
class TraditionalDLModels(nn.Module):
    def __init__(self, in_dim=30, num_classes=5, model_type="lstm", hidden_dim=128, num_layers=2):
        super().__init__()
        self.model_type = model_type
        if model_type == "lstm":
            self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_classes))
        elif model_type == "cnn":
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten()
            )
            self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_classes))
    def forward(self, x):
        if self.model_type == "lstm":
            x = x.unsqueeze(1)
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)
        elif self.model_type == "cnn":
            x = x.unsqueeze(1)
            out = self.cnn(x)
            return self.fc(out)

# Evaluation utilities
def detailed_metrics(y_true, y_pred, class_names=None):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    accuracy = np.mean(y_true == y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # FPR, FNR, Specificity
    fpr, fnr, specificity = [], [], []
    for i in range(len(cm)):
        tn = np.sum(cm) - np.sum(cm[i]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        fn = np.sum(cm[i]) - cm[i, i]
        tp = cm[i, i]
        fpr_i = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr_i = fn / (fn + tp) if (fn + tp) > 0 else 0
        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr.append(fpr_i); fnr.append(fnr_i); specificity.append(spec_i)

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

def generate_and_save_classification_report(y_true, y_pred, class_names, experiment_name, timestamp):
    # Generate and save classification report with per-class accuracy
    report_text = classification_report(y_true, y_pred,
                                        target_names=class_names,
                                        digits=4)
    print("\n[Classification Report]")
    print(report_text)

    report_dict = classification_report(y_true, y_pred,
                                        target_names=class_names,
                                        digits=4,
                                        output_dict=True)
    df = pd.DataFrame(report_dict).T

    # Calculate per-class One-vs-Rest accuracy
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    N = cm.sum()
    per_class_accuracy = []
    for i in labels:
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = N - TP - FP - FN
        acc_i = (TP + TN) / N if N > 0 else 0.0
        per_class_accuracy.append(acc_i)

    # Print per-class accuracy
    print("\n[Per-class Accuracy (One-vs-Rest)]")
    header = f"{'class':<14}{'accuracy':>12}{'support':>12}"
    print(header)
    print("-" * len(header))
    for i, cname in enumerate(class_names):
        support_i = int(cm[i, :].sum())
        print(f"{cname:<14}{per_class_accuracy[i]:>12.4f}{support_i:>12d}")

    # Update DataFrame
    for cname, acc_i in zip(class_names, per_class_accuracy):
        if cname in df.index:
            df.loc[cname, 'class_accuracy'] = acc_i

    # Save to TXT
    txt_path = f"classification_report_{experiment_name}_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("[Classification Report]\n")
        f.write(report_text + "\n\n")
        f.write("[Per-class Accuracy (One-vs-Rest)]\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for i, cname in enumerate(class_names):
            support_i = int(cm[i, :].sum())
            f.write(f"{cname:<14}{per_class_accuracy[i]:>12.4f}{support_i:>12d}\n")
    print(f"Classification report (txt) saved: {txt_path}")

    # Save to CSV
    csv_path = f"classification_report_{experiment_name}_{timestamp}.csv"
    df.to_csv(csv_path, encoding='utf-8-sig')
    print(f"Classification report (csv) saved: {csv_path}")

# Training utilities
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
        loss = criterion(logits, y) if criterion is not None else F.cross_entropy(logits, y)
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

# Single model experiment
def run_single_dl_experiment(model_type, experiment_name, train_loader, val_loader, test_loader):
    print(f"\n{'=' * 60}")
    print(f"Starting deep learning experiment: {experiment_name} ({model_type})")
    print(f"{'=' * 60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = TraditionalDLModels(
        in_dim=NUM_FEATURES, model_type=model_type, hidden_dim=128, num_layers=2
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

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
    print(f"Training completed, time taken: {training_time:.2f} seconds")

    print("Starting test set evaluation...")
    model.load_state_dict(torch.load(f"best_model_{experiment_name}.pt", weights_only=True))

    test_loss, test_acc, test_y, test_pred = run_epoch(test_loader, train=False, model=model, criterion=criterion,
                                                       device=device)

    # Get probabilities for AUC calculations
    model.eval()
    all_probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    y_prob = np.concatenate(all_probs, axis=0)

    # Calculate metrics
    class_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
    test_metrics = detailed_metrics(test_y, test_pred, class_names=class_names)
    test_metrics['roc_auc'] = roc_auc_score(test_y, y_prob, multi_class='ovr', average='weighted')
    test_metrics['pr_auc'] = average_precision_score(test_y, y_prob, average='weighted')

    # Generate and save classification report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    generate_and_save_classification_report(test_y, test_pred, class_names, experiment_name, timestamp)

    # Check prediction distributions
    unique, counts = np.unique(test_pred, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique, counts))}")
    print(f"True distribution: {dict(zip(*np.unique(test_y, return_counts=True)))}")
    recalculated_accuracy = (np.array(test_y) == np.array(test_pred)).mean()
    print(f"Recalculated Accuracy: {recalculated_accuracy:.4f}")

    # Compile results
    results = {
        'experiment_name': experiment_name,
        'model_type': model_type,
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

    print(f"\n{experiment_name} test results:")
    print(f"Accuracy: {results['test_accuracy']:.4f}")
    print(f"Precision: {results['test_precision']:.4f}")
    print(f"Recall: {results['test_recall']:.4f}")
    print(f"F1-Score: {results['test_f1']:.4f}")
    print(f"ROC-AUC: {results['test_roc_auc']:.4f}")
    print(f"PR-AUC: {results['test_pr_auc']:.4f}")
    print(f"MCC: {results['test_mcc']:.4f}")
    print(f"Balanced Accuracy: {results['test_balanced_accuracy']:.4f}")

    return results

# Batch deep learning model comparison experiments
def run_all_dl_comparison_experiments():
    dl_configs = [
        ("lstm", "LSTM"),
        ("cnn", "CNN")
    ]
    all_results = []
    print("Starting batch deep learning model comparison experiments...")
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}, Test set size: {len(X_test)}")
    print(f"Feature dimension: {NUM_FEATURES}")

    for model_type, experiment_name in dl_configs:
        try:
            result = run_single_dl_experiment(model_type, experiment_name, train_loader, val_loader, test_loader)
            all_results.append(result)
            print(f"Experiment {experiment_name} completed successfully")
        except Exception as e:
            print(f"Experiment {experiment_name} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 160}")
    print("Traditional deep learning model comparison results")
    print(f"{'=' * 160}")
    print(
        f"{'Model Name':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} "
        f"{'ROC-AUC':<10} {'PR-AUC':<10} {'MCC':<10} {'Balanced Acc':<12} "
        f"{'Weighted FPR':<12} {'Weighted FNR':<12} {'Training Time(s)':<12}")
    print("-" * 160)

    for result in all_results:
        print(f"{result['experiment_name']:<15} "
              f"{result['test_accuracy']:<10.4f} "
              f"{result['test_precision']:<10.4f} "
              f"{result['test_recall']:<10.4f} "
              f"{result['test_f1']:<10.4f} "
              f"{result['test_roc_auc']:<10.4f} "
              f"{result['test_pr_auc']:<10.4f} "
              f"{result['test_mcc']:<10.4f} "
              f"{result['test_balanced_accuracy']:<12.4f} "
              f"{result['test_weighted_fpr']:<10.4f} "
              f"{result['test_weighted_fnr']:<10.4f} "
              f"{result['training_time']:<12.1f}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f'dl_comparison_results_{timestamp}.json'
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {results_filename}")

    plot_dl_comparison_results(all_results, timestamp)
    return all_results

def plot_dl_comparison_results(results, timestamp):
    if not results:
        print("No results to plot")
        return
    experiment_names = [r['experiment_name'] for r in results]
    metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1',
               'test_roc_auc', 'test_pr_auc', 'test_mcc', 'test_balanced_accuracy']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'MCC', 'Balanced Acc']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange', 'lightseagreen', 'lightsalmon']
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [r[metric] for r in results]
        bars = axes[i].bar(range(len(experiment_names)), values, color=colors[i])
        axes[i].set_title(metric_name, fontsize=14, pad=20)
        axes[i].set_xticks(range(len(experiment_names)))
        axes[i].set_xticklabels(experiment_names, rotation=45, ha='right', fontsize=10)
        axes[i].set_ylim(0, 1); axes[i].grid(axis='y', alpha=0.3)
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plot_filename = f'dl_comparison_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison plot saved to: {plot_filename}")

# Model complexity analysis
def analyze_dl_model_complexity():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_info = {
        "LSTM": TraditionalDLModels(NUM_FEATURES, model_type="lstm", hidden_dim=128, num_layers=2),
        "CNN": TraditionalDLModels(NUM_FEATURES, model_type="cnn"),
    }
    print(f"\n{'=' * 80}")
    print("Deep learning model complexity analysis")
    print(f"{'=' * 80}")
    print(f"{'Model Name':<15} {'Parameter Count':<15} {'Estimated Size(MB)':<15}")
    print("-" * 80)
    for name, model in models_info.items():
        model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * 4 / (1024 * 1024)
        print(f"{name:<15} {param_count:<15,} {model_size_mb:<15.2f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Performance comparison analysis
def analyze_dl_performance_comparison(results):
    if not results:
        print("No results to analyze")
        return
    print(f"\n{'=' * 60}")
    print("Deep learning model performance comparison analysis")
    print(f"{'=' * 60}")
    best_result = max(results, key=lambda x: x['test_f1'])
    print(f"Best model: {best_result['experiment_name']}")
    print(f"   Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"   F1-Score: {best_result['test_f1']:.4f}")
    print(f"   ROC-AUC: {best_result['test_roc_auc']:.4f}")
    print(f"   MCC: {best_result['test_mcc']:.4f}")
    print(f"\nModel performance comparison:")
    print(f"{'Model':<12} {'Accuracy':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'MCC':<10}")
    print("-" * 50)
    for result in results:
        print(f"{result['experiment_name']:<12} "
              f"{result['test_accuracy']:<10.4f} "
              f"{result['test_f1']:<10.4f} "
              f"{result['test_roc_auc']:<10.4f} "
              f"{result['test_mcc']:<10.4f}")

# Main function
def main():
    print("Traditional deep learning model comparison experiment")
    print("=" * 80)
    print("Experiment configuration:")
    print(f"  - Dataset path: {root}")
    print(f"  - Feature dimension: {NUM_FEATURES}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Model parameters: hidden_dim=128, num_layers=2")
    print("=" * 80)

    analyze_dl_model_complexity()
    print("\nStarting deep learning model comparison experiments...")
    results = run_all_dl_comparison_experiments()

    if results:
        analyze_dl_performance_comparison(results)
        print(f"\n{'=' * 60}")
        print("Experiment summary")
        print(f"{'=' * 60}")
        best_result = max(results, key=lambda x: x['test_f1'])
        print(f"Best model: {best_result['experiment_name']}")
        print(f"   F1-Score: {best_result['test_f1']:.4f}")
        print(f"   ROC-AUC: {best_result['test_roc_auc']:.4f}")
        print(f"   MCC: {best_result['test_mcc']:.4f}")
        print(f"\nNote: Classification reports for each model have been printed and saved as TXT/CSV.")

    print(f"\nDeep learning comparison experiment completed!")

# Entry point
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        if model_type in ["lstm", "cnn"]:
            result = run_single_dl_experiment(
                model_type,
                {"lstm": "LSTM",
                 "cnn": "CNN"}[model_type],
                train_loader, val_loader, test_loader
            )
            print(f"Single experiment result: {result}")
        else:
            print("Usage: python dl_comparison_experiment.py [lstm|cnn]")
    else:
        main()