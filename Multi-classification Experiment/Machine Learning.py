import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_curve, auc,roc_auc_score, precision_recall_curve,
    confusion_matrix, precision_score, recall_score,
    f1_score, matthews_corrcoef, balanced_accuracy_score,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import random
import os

RANDOM_SEED = 3407

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

try:
    rcParams['font.sans-serif'] = ['Microsoft YaHei']
except Exception:
    rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 14
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Read data
# =============================================================================
TRAIN_PATH = r"...SCRM_train_mix.csv"
TEST_PATH = r"...SCRM_test_mix.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns='SCMstability_category')
y_train = train_df['SCMstability_category']
X_test = test_df.drop(columns='SCMstability_category')
y_test = test_df['SCMstability_category']

# =============================================================================
# 2. Model Definition
# =============================================================================
MODELS = {
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=5, random_state=42, n_jobs=-1
    ),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, random_state=42, n_jobs=-1
    ),
    "NaiveBayes": GaussianNB(),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=200, max_depth=5, random_state=42, n_jobs=-1
    )
}


# =============================================================================
# 3. Evaluation Indicators
# =============================================================================
def calculate_comprehensive_metrics(y_true, y_pred, y_prob=None, class_names=None):
    if class_names is None:
        classes_sorted = np.unique(y_true)
        class_names = [f"Class {c}" for c in classes_sorted]
    else:
        classes_sorted = np.unique(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=classes_sorted)
    fpr_list, fnr_list = [], []
    for i in range(len(cm)):
        tn = np.sum(cm) - np.sum(cm[i]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        fn = np.sum(cm[i]) - cm[i, i]
        tp = cm[i, i]
        fpr_i = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr_i = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr_list.append(fpr_i); fnr_list.append(fnr_i)
    macro_fpr = np.mean(fpr_list); macro_fnr = np.mean(fnr_list)
    weighted_fpr = np.average(fpr_list, weights=[np.sum(cm[i]) for i in range(len(cm))])
    weighted_fnr = np.average(fnr_list, weights=[np.sum(cm[i]) for i in range(len(cm))])

    roc_auc, pr_auc = None, None
    if y_prob is not None:
        if y_prob.ndim == 1:
            y_prob = np.stack([1 - y_prob, y_prob], axis=1)

        row_sum = y_prob.sum(axis=1, keepdims=True)
        y_prob = y_prob / (row_sum + 1e-12)

        n_cols = y_prob.shape[1]
        n_classes = len(classes_sorted)

        if n_cols != n_classes:
            keep_idx = [int(c) for c in classes_sorted if int(c) < n_cols]
            y_prob = y_prob[:, keep_idx]
            n_cols = y_prob.shape[1]

        if n_classes == 2 and n_cols == 2:
            pos_col = 1
            roc_auc = roc_auc_score(y_true, y_prob[:, pos_col])
            pr_auc  = average_precision_score(y_true, y_prob[:, pos_col])
        elif n_classes >= 2 and n_cols == n_classes:
            roc_auc = roc_auc_score(
                y_true,
                y_prob,
                multi_class='ovr',
                average='weighted',
                labels=classes_sorted
            )

            y_true_bin = label_binarize(y_true, classes=classes_sorted)
            pr_auc = average_precision_score(
                y_true_bin,
                y_prob,
                average='weighted'
            )

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'balanced_accuracy': balanced_acc,
        'macro_fpr': macro_fpr,
        'macro_fnr': macro_fnr,
        'weighted_fpr': weighted_fpr,
        'weighted_fnr': weighted_fnr,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist()
    }
    for i, name in enumerate(class_names[:len(fpr_list)]):
        metrics[f'{name}_fpr'] = fpr_list[i]
        metrics[f'{name}_fnr'] = fnr_list[i]
    return metrics

def generate_and_save_classification_report(y_true, y_pred, class_names, experiment_name, timestamp):
    """Generate and save classification reports (TXT/CSV)"""
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

    print("\n[Per-class Accuracy (One-vs-Rest)]")
    header = f"{'class':<18}{'accuracy':>12}{'support':>12}"
    print(header)
    print("-" * len(header))
    for i, cname in enumerate(class_names):
        support_i = int(cm[i, :].sum())
        print(f"{cname:<18}{per_class_accuracy[i]:>12.4f}{support_i:>12d}")

    for cname, acc_i in zip(class_names, per_class_accuracy):
        if cname in df.index:
            df.loc[cname, 'class_accuracy'] = acc_i

    txt_path = f"classification_report_{experiment_name}_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("[Classification Report]\n")
        f.write(report_text + "\n\n")
        f.write("[Per-class Accuracy (One-vs-Rest)]\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for i, cname in enumerate(class_names):
            support_i = int(cm[i, :].sum())
            f.write(f"{cname:<18}{per_class_accuracy[i]:>12.4f}{support_i:>12d}\n")
    print(f"Classification report (txt) saved: {txt_path}")

    csv_path = f"classification_report_{experiment_name}_{timestamp}.csv"
    df.to_csv(csv_path, encoding='utf-8-sig')
    print(f"Classification report (csv) saved: {csv_path}")


# =============================================================================
# 5. Training & Evaluation
# =============================================================================
results = {}
classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = len(classes)
class_names = [str(c) for c in classes]

for name, model in MODELS.items():
    print(f"\n{'=' * 60}")
    print(f"Begin training {name} ...")
    tic = time.perf_counter()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_prob = None
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_test)
        except Exception as e:
            print(f"Warning: Model {name} predict_proba failed: {e}")
            y_prob = None

    metrics = calculate_comprehensive_metrics(y_test, y_pred, y_prob, class_names)

    accuracy = metrics['accuracy']
    report = classification_report(y_test, y_pred, digits=4)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': accuracy,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'mcc': metrics['mcc'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'macro_fpr': metrics['macro_fpr'],
        'macro_fnr': metrics['macro_fnr'],
        'weighted_fpr': metrics['weighted_fpr'],
        'weighted_fnr': metrics['weighted_fnr'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc'],
        'report': report,
        'train_time': time.perf_counter() - tic,
        'confusion_matrix': metrics['confusion_matrix']
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = name
    generate_and_save_classification_report(y_test, y_pred, class_names, experiment_name, timestamp)

    print(report)
    print(f"Accuracy: {accuracy:.4f} | Training time: {results[name]['train_time']:.2f}s")

    print(f"\nComprehensive Evaluation Indicators:")
    print(f"  Accuracy: {results[name]['accuracy']:.4f}")
    print(f"  Precision: {results[name]['precision']:.4f}")
    print(f"  Recall: {results[name]['recall']:.4f}")
    print(f"  F1-Score: {results[name]['f1']:.4f}")
    print(f"  MCC: {results[name]['mcc']:.4f}")
    print(f"  Balanced Accuracy: {results[name]['balanced_accuracy']:.4f}")
    print(f"  Macro FPR: {results[name]['macro_fpr']:.4f}")
    print(f"  Macro FNR: {results[name]['macro_fnr']:.4f}")
    print(f"  Weighted FPR: {results[name]['weighted_fpr']:.4f}")
    print(f"  Weighted FNR: {results[name]['weighted_fnr']:.4f}")
    if results[name]['roc_auc'] is not None:
        print(f"  ROC-AUC: {results[name]['roc_auc']:.4f}")
    if results[name]['pr_auc'] is not None:
        print(f"  PR-AUC: {results[name]['pr_auc']:.4f}")

    for i, class_name in enumerate(class_names):
        if f'{class_name}_fpr' in results[name]:
            print(f"  {class_name} FPR: {results[name][f'{class_name}_fpr']:.4f}")
            print(f"  {class_name} FNR: {results[name][f'{class_name}_fnr']:.4f}")

    print('=' * 60)


# =============================================================================
# 6. Visualization function
# =============================================================================
def plot_accuracy_bar(results_dict: dict, save_path: str = 'accuracy_bar.png'):
    """Plot bar charts showing the accuracy rates of each model"""
    names = list(results_dict.keys())
    accs = [results_dict[m]['accuracy'] for m in names]

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))
    sns.barplot(x=accs, y=names, palette='viridis')
    plt.title('Comparison of Accuracy Rates Across Models', fontsize=16)
    plt.xlabel('Accuracy', fontsize=14)
    plt.ylabel('Model', fontsize=14)

    for i, v in enumerate(accs):
        plt.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_comprehensive_metrics_bar(results_dict: dict, save_path: str = 'comprehensive_metrics.png'):
    """Create a comprehensive indicator comparison chart"""
    names = list(results_dict.keys())

    # Extract various indicators
    metrics_data = {
        'Accuracy': [results_dict[m]['accuracy'] for m in names],
        'Precision': [results_dict[m]['precision'] for m in names],
        'Recall': [results_dict[m]['recall'] for m in names],
        'F1-Score': [results_dict[m]['f1'] for m in names],
        'MCC': [results_dict[m]['mcc'] for m in names],
        'Balanced Acc': [results_dict[m]['balanced_accuracy'] for m in names]
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange']

    for i, (metric_name, values) in enumerate(metrics_data.items()):
        bars = axes[i].bar(range(len(names)), values, color=colors[i])
        axes[i].set_title(metric_name, fontsize=14)
        axes[i].set_xticks(range(len(names)))
        axes[i].set_xticklabels(names, rotation=45, ha='right')
        axes[i].set_ylim(0, 1)
        axes[i].grid(axis='y', alpha=0.3)

        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_auc(results_dict: dict, class_names, save_path: str = 'roc_auc_curve.png'):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    classes_sorted = np.array([int(c) if str(c).isdigit() else c for c in class_names])
    plt.figure(figsize=(8, 6))
    plotted = False

    for name, res in results_dict.items():
        y_prob = res['y_prob']
        if y_prob is None:
            print(f"Warning: Model {name} has no prediction probabilities; skipping ROC."); continue


        if y_prob.ndim == 1: y_prob = np.stack([1-y_prob, y_prob], axis=1)
        row_sum = y_prob.sum(axis=1, keepdims=True); y_prob = y_prob / (row_sum + 1e-12)
        n_cols = y_prob.shape[1]; n_classes = len(classes_sorted)
        if n_cols != n_classes:
            keep_idx = [int(c) for c in classes_sorted if isinstance(c, (int, np.integer)) and int(c) < n_cols]
            y_prob = y_prob[:, keep_idx]


        y_true_bin = label_binarize(y_test, classes=np.unique(y_test))
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob[:, :y_true_bin.shape[1]].ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.8, label=f'{name} (micro AUC={roc_auc:.3f})')
        plotted = True

    if not plotted:
        print("There is no ROC data available for plotting."); return

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False positive rate'); plt.ylabel('True Rate'); plt.title('Multi-class ROC（micro-average）')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True); plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.show()



def plot_pr_curve(results_dict: dict, class_names, save_path: str = 'pr_curve.png'):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 6))
    plotted = False
    classes_sorted = np.unique(y_test)
    y_true_bin = label_binarize(y_test, classes=classes_sorted)

    for name, res in results_dict.items():
        y_prob = res['y_prob']
        if y_prob is None:
            print(f"Warning: Model {name} has no prediction probabilities; skipping PR."); continue

        if y_prob.ndim == 1: y_prob = np.stack([1-y_prob, y_prob], axis=1)
        row_sum = y_prob.sum(axis=1, keepdims=True); y_prob = y_prob / (row_sum + 1e-12)
        if y_prob.shape[1] != y_true_bin.shape[1]:
            keep_idx = [int(c) for c in classes_sorted if int(c) < y_prob.shape[1]]
            y_prob = y_prob[:, keep_idx]

        precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_prob[:, :y_true_bin.shape[1]].ravel())
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=1.8, label=f'{name} (micro AP={pr_auc:.3f})')
        plotted = True

    if not plotted:
        print("No PR data available for plotting."); return

    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Multi-class PR（micro-average）')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True); plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.show()



def plot_confusion_matrix(results_dict: dict, class_names, save_path: str = 'confusion_matrix.png'):
    """Plot the confusion matrix for all models"""

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    n_models = len(results_dict)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))

    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    for idx, (name, res) in enumerate(results_dict.items(), 1):
        cm = np.array(res['confusion_matrix'])
        plt.subplot(n_rows, n_cols, idx)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=class_names, yticklabels=class_names)

        plt.title(f'{name} Confusion Matrix', fontsize=14)
        plt.xlabel('Prediction Label', fontsize=12)
        plt.ylabel('Authenticity Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_comprehensive_comparison(results_dict: dict):
    """Model Comparison"""
    print(f"\n{'=' * 160}")
    print("Comprehensive Performance Comparison of Models")
    print(f"{'=' * 160}")
    print(
        f"{'Model Name':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'PR-AUC':<10} {'MCC':<10} {'Balanced Acc':<12} {'Weighted FPR':<12} {'Weighted FNR':<12} {'Training Time(s)':<12}")
    print("-" * 160)

    for name, result in results_dict.items():
        print(f"{name:<15} "
              f"{result['accuracy']:<10.4f} "
              f"{result['precision']:<10.4f} "
              f"{result['recall']:<10.4f} "
              f"{result['f1']:<10.4f} "
              f"{result['roc_auc'] if result['roc_auc'] is not None else 0:<10.4f} "
              f"{result['pr_auc'] if result['pr_auc'] is not None else 0:<10.4f} "
              f"{result['mcc']:<10.4f} "
              f"{result['balanced_accuracy']:<12.4f} "
              f"{result['weighted_fpr']:<12.4f} "  
              f"{result['weighted_fnr']:<12.4f} "  
              f"{result['train_time']:<12.1f}")


# =============================================================================
# 7. Generate all graphics and reports
# =============================================================================
if __name__ == '__main__':
    print_comprehensive_comparison(results)
    plot_accuracy_bar(results)
    plot_comprehensive_metrics_bar(results)
    plot_roc_auc(results, class_names)
    plot_pr_curve(results, class_names)
    plot_confusion_matrix(results, class_names)
