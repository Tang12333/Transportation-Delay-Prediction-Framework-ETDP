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
import matplotlib.pyplot as plt
import random
import os
import shap
import time

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
val_csv   = root / "val_scaled_binary.csv"
test_csv  = root / "test_scaled_binary.csv"

# =========================================================
# 2. Read & Partition
# =========================================================
def load_xy(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['delivery_risk']).values.astype('float32')
    y = df['delivery_risk'].values.astype('int64')
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
class LGFT(nn.Module):
    def __init__(self, in_dim=30, num_classes=2, dim=128, heads=4, depth=3):
        super().__init__()
        # Local CNN
        self.local_path = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1, groups=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, dim, 3, padding=1, groups=4),
            nn.BatchNorm1d(dim), nn.ReLU()
        )
        # Global MLP
        self.global_path = nn.Sequential(
            nn.Linear(in_dim, dim * 2), nn.ReLU(),
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, in_dim, dim))
        # Transformer
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
    specificity = [(np.sum(cm) - np.sum(cm[i]) - np.sum(cm[:, i]) + cm[i, i]) /
                   (np.sum(cm) - np.sum(cm[i]) - np.sum(cm[:, i]) + cm[i, i] + (np.sum(cm[:, i]) - cm[i, i]))
                   if (np.sum(cm) - np.sum(cm[i]) - np.sum(cm[:, i]) + cm[i, i] + (np.sum(cm[:, i]) - cm[i, i])) > 0
                   else 0 for i in range(len(cm))]
    metrics = {
        'accuracy': accuracy,
        'macro_precision': np.mean(precision), 'macro_recall': np.mean(recall),
        'macro_f1': np.mean(f1), 'macro_specificity': np.mean(specificity),
        'weighted_precision': np.average(precision, weights=support),
        'weighted_recall': np.average(recall, weights=support),
        'weighted_f1': np.average(f1, weights=support),
        'weighted_specificity': np.average(specificity, weights=support),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'roc_auc': None,
        'pr_auc': None,
        'confusion_matrix': cm.tolist()
    }
    for i, name in enumerate(class_names):
        metrics[f'{name}_precision'], metrics[f'{name}_recall'] = precision[i], recall[i]
        metrics[f'{name}_f1'], metrics[f'{name}_specificity'] = f1[i], specificity[i]
        metrics[f'{name}_support'] = support[i]
    return metrics

# =========================================================
# 5. Drawing Tools
# =========================================================
def plot_roc_curves(y_true, y_prob, class_names=['OnTime', 'Delay']):
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{class_names[1]} (AUC={auc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False positive rate', fontsize=12)
    plt.ylabel('True Rate', fontsize=12)
    plt.title('ROC curve', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300)
    plt.show()

def plot_pr_curves(y_true, y_prob, class_names):
    plt.figure(figsize=(6, 5))
    prec, rec, _ = precision_recall_curve(y_true, y_prob[:, 1])
    ap = average_precision_score(y_true, y_prob[:, 1])
    plt.plot(rec, prec, lw=2, label=f'{class_names[1]} (AP={ap:.3f})')
    plt.xlabel('Recall rate', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('PR curve', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pr_curve.png', dpi=300)
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
    plt.xlabel('Prediction label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

# =========================================================
# 6. Loss & Training
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
model = LGFT(in_dim=NUM_FEATURES, dim=128, heads=4, depth=3).to(device)
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

class_names = ['On time', 'Late']
test_metrics = detailed_metrics(test_y, test_pred, class_names=class_names)

test_metrics['roc_auc'] = roc_auc_score(test_y, y_prob[:, 1])
test_metrics['pr_auc'] = average_precision_score(test_y, y_prob[:, 1])

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

# =========================================================
# 8. SHAP
# =========================================================
def add_shap_analysis(model, test_loader, train_loader, device, class_names, feature_names=None, max_eval_samples=100,
                      max_vis_samples=20):

    print(f"{'=' * 50}")
    print("SHAP Explainability Analysis Begins")
    print(f"{'=' * 50}")

    # --- 1. SHAP Explainability Analysis Begins ---
    # a. Acquire test samples for interpretation
    test_samples_for_vis = []
    test_labels_for_vis = []
    test_samples_count = 0
    for x_batch, y_batch in test_loader:
        for i in range(len(x_batch)):
            if test_samples_count < max_vis_samples:
                test_samples_for_vis.append(x_batch[i].unsqueeze(0))
                test_labels_for_vis.append(y_batch[i].item())
                test_samples_count += 1
            else:
                break
        if test_samples_count >= max_vis_samples:
            break
    X_test_shap_vis = torch.cat(test_samples_for_vis, dim=0)
    print(f"Select {len(test_samples_for_vis)} test samples for SHAP visualization.")
    print("Retrieving prediction results for samples used in visualization...")
    model.eval()
    test_pred_for_vis = []
    with torch.no_grad():
        for x in test_samples_for_vis:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(1).cpu().item()
            test_pred_for_vis.append(pred)
    print(f"Prediction results for {len(test_pred_for_vis)} samples have been obtained.")

    # b. Prepare background data
    background_samples = []
    background_count = 0
    background_samples_needed = min(len(X_test_shap_vis), max_eval_samples - len(X_test_shap_vis))
    if background_samples_needed <= 0:
        print("Warning: If max_eval_samples is less than or equal to the number of visual samples, no additional background samples will be used.")
        background_samples_needed = 0
    for x_batch, _ in train_loader:
        for i in range(len(x_batch)):
            if background_count < background_samples_needed:
                background_samples.append(x_batch[i].unsqueeze(0))
                background_count += 1
            else:
                break
        if background_count >= background_samples_needed:
            break
    if background_samples:
        background_data = torch.cat(background_samples, dim=0)
        print(f"Extract {len(background_samples)} samples from the training set as background data.")
    else:
        num_bg_from_test = min(10, len(X_test_shap_vis))
        background_data = X_test_shap_vis[:num_bg_from_test]
        print(f"Using {num_bg_from_test} test samples as background data (due to constraints).")

    # Move to CPU for SHAP computation
    background_data = background_data.cpu()
    X_test_shap_vis = X_test_shap_vis.cpu()
    original_device = next(model.parameters()).device
    model = model.cpu()
    print(f"Background data shape: {background_data.shape}, Device: {background_data.device}")
    print(f"Visualization test data shape: {X_test_shap_vis.shape}, Device: {X_test_shap_vis.device}")
    print(f"The model has been moved to the device: {next(model.parameters()).device}")

    # --- 2. Define the PyTorch model prediction function ---
    def model_predict_fn(data_tensor):
        """The model prediction function for SHAP calls, returning the probabilities for each category."""
        if isinstance(data_tensor, np.ndarray):
            data_tensor = torch.from_numpy(data_tensor).float()
        data_tensor = data_tensor.cpu()
        model.eval()
        with torch.no_grad():
            logits = model(data_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # --- 3. Create SHAP interpreters ---
    print("Creating SHAP GradientExplainer...")
    start_time = time.time()
    explainer = shap.GradientExplainer(model, background_data)
    print(f"SHAP Explainer creation completed in {time.time() - start_time:.2f} seconds.")

    # --- 4. Calculate SHAP values ---
    print("Calculating SHAP values...")
    start_time = time.time()
    X_test_shap_vis_cpu = X_test_shap_vis
    shap_values = None
    try:
        shap_values = explainer.shap_values(X_test_shap_vis_cpu)
        print("Compute SHAP values using the default settings.")
    except Exception as e1:
        print(f"Failed to use default settings: {e1}")
        try:
            print("Try using nsamples=50...")
            shap_values = explainer.shap_values(X_test_shap_vis_cpu, nsamples=50)
        except Exception as e2:
            print(f"Using nsamples=50 also failed: {e2}")
            print("Try using nsamples=‘auto’...")
            try:
                shap_values = explainer.shap_values(X_test_shap_vis_cpu, nsamples='auto')
            except Exception as e3:
                print(f"Using nsamples=‘auto’ also fails: {e3}")
                print("SHAP value calculation failed.")
    if shap_values is None:
        print("Failed to compute SHAP values; subsequent visualization skipped.")
        return
    print(f"SHAP value calculation completed in {time.time() - start_time:.2f} seconds.")

    # SHAP value calculation completed in {time.time() - start_time:.2f} seconds.
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        print(f"SHAP value structure: Single numpy.ndarray, shape: {shap_values.shape}")
        N, D, C = shap_values.shape
        shap_values_per_class = [shap_values[:, :, i] for i in range(C)]
        shap_values = shap_values_per_class
        print(f"Converted to a list containing {len(shap_values)} category arrays, each with shape {shap_values[0].shape}")
    elif isinstance(shap_values, list):
        print(f"SHAP value structure: list, list length (number of categories): {len(shap_values)}")
        if len(shap_values) > 0:
            print(f"First category: SHAP value shape: {shap_values[0].shape}")
    else:
        print(f"Warning: SHAP values are of unexpected type: {type(shap_values)} or shape. Attempting conversion...")
        try:
            shap_values = [np.array(shap_values)]
            print(f"After conversion, list length: {len(shap_values)}, element shape: {shap_values[0].shape}")
        except Exception as convert_e:
            print(f"Conversion failed: {convert_e}. Unable to process SHAP values; skipping subsequent visualization.")
            return

    # Calculate expected_value - Adapted for binary classification
    print("Calculating the baseline expected value for each category...")
    try:
        background_probs = model_predict_fn(background_data.numpy())
        expected_value = np.mean(background_probs, axis=0)
        print(f"The computed expected_value: {expected_value} (shape: {expected_value.shape})")
    except Exception as e:
        print(f"Failed to compute expected_value: {e}. Default value [0, 0] will be used.")
        expected_value = np.array([0.0, 0.0])

    # --- 5. Prepare Feature Name ---
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X_test_shap_vis.shape[1])]
    print(f"Number of feature names used: {len(feature_names)}")

    # --- 6. Visualization ---
    # a. SHAP Summary Plot - Adapted for binary classification
    print("生成 SHAP Summary Plot ...")
    try:
        for i, class_name in enumerate(class_names):
            if i < len(shap_values) and isinstance(shap_values[i], np.ndarray) and shap_values[i].ndim == 2:
                plt.figure(figsize=(10, 8))
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
                plt.rcParams['axes.unicode_minus'] = False
                shap.summary_plot(shap_values[i], X_test_shap_vis_cpu.numpy(),
                                  feature_names=feature_names,
                                  show=False, plot_size=None, plot_type="dot")
                plt.title(f"SHAP Summary Plot for {class_name}")
                plt.tight_layout()
                plt.savefig(f'shap_summary_{class_name}.png', dpi=150, bbox_inches='tight')
                plt.show()
                print(f"  - Summary Plot for {class_name} saved: shap_summary_{class_name}.png")
            else:
                print(f"  - Warning: SHAP values for category {i} are in an invalid format and have been skipped.")
    except Exception as e:
        print(f"Error generating Summary Plot: {e}")
        import traceback
        traceback.print_exc()

    # b. SHAP Dependence Plot
    print("Generate SHAP Dependence Plot...")
    try:
        main_feature = "order_processing_days"
        for i, class_name in enumerate(class_names):
            if i < len(shap_values) and isinstance(shap_values[i], np.ndarray):
                plt.figure(figsize=(8, 6))
                shap.dependence_plot(
                    main_feature,
                    shap_values[i],
                    X_test_shap_vis_cpu.numpy(),
                    feature_names=feature_names,
                    interaction_index="shipping_mode",
                    show=False
                )
                plt.title(f"{main_feature} → {class_name} (by shipping_mode)")
                plt.tight_layout()
                plt.savefig(f'shap_dependence_{main_feature}_vs_{class_name}.png', dpi=150, bbox_inches='tight')
                plt.show()
    except Exception as e:
        print(f"Error occurred while generating the Dependence Plot: {e}")
        import traceback
        traceback.print_exc()

    # c. SHAP Decision Plot (Global Decision Plot) - Modified to generate for both categories
    print("Generate SHAP Decision Plot (Two Categories)...")
    try:
        for i, class_name in enumerate(class_names):
            if i >= len(shap_values) or not isinstance(shap_values[i], np.ndarray):
                print(f"  - Skip class {class_name}; SHAP values are unavailable.")
                continue

            global_shap_vals = shap_values[i]
            global_feature_vals = X_test_shap_vis_cpu.numpy()

            # Create a global interpretation
            global_explanation = shap.Explanation(
                values=global_shap_vals,
                base_values=np.full(global_shap_vals.shape[0], expected_value[i]),
                data=global_feature_vals,
                feature_names=feature_names
            )

            plt.figure(figsize=(14, 10))
            shap.decision_plot(
                expected_value[i],
                global_shap_vals,
                global_feature_vals,
                feature_names=feature_names,
                feature_order='importance',
                highlight=test_labels_for_vis,
                show=False,
                legend_labels=class_names,
                legend_location='best'
            )
            plt.title(f"SHAP Decision Plot for '{class_name}'")
            plt.tight_layout()
            plt.savefig(f'shap_decision_plot_{class_name}.png', dpi=150, bbox_inches='tight')
            plt.show()
            print(f"  - Decision Plot for {class_name} saved: shap_decision_plot_{class_name}.png")
    except Exception as e:
        print(f"Error occurred while generating the Decision Plot: {e}")
        import traceback
        traceback.print_exc()

    # d. SHAP Waterfall Plot (Single Sample) - Optimized Title
    print("Generate SHAP Waterfall Plot (Sample Example)...")
    try:
        sample_idx = 0
        if test_labels_for_vis and len(test_labels_for_vis) > sample_idx:
            true_label_name = class_names[test_labels_for_vis[sample_idx]]
        else:
            true_label_name = "Unknown"
        sample_probs = model_predict_fn(X_test_shap_vis_cpu[sample_idx:sample_idx + 1])
        pred_label_idx = np.argmax(sample_probs[0])
        pred_label_name = class_names[pred_label_idx]

        if pred_label_idx < len(shap_values) and isinstance(shap_values[pred_label_idx], np.ndarray):
            shap_val_for_sample = shap_values[pred_label_idx][sample_idx]
            feature_values_for_sample = X_test_shap_vis_cpu[sample_idx].numpy()
            base_value = expected_value[pred_label_idx]

            explanation = shap.Explanation(
                values=shap_val_for_sample,
                base_values=base_value,
                data=feature_values_for_sample,
                feature_names=feature_names
            )

            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(explanation, show=False, max_display=15)
            # Simplified Title
            plt.title(f"Waterfall: Sample {sample_idx} | True={true_label_name}, Pred={pred_label_name}")
            plt.tight_layout()
            plt.savefig(f'shap_waterfall_sample_{sample_idx}_pred_{pred_label_name}.png', dpi=150, bbox_inches='tight')
            plt.show()
            print(f"  - Waterfall Plot for saved sample {sample_idx}.")
        else:
            print(f"  - Error: Unable to obtain SHAP values for sample {sample_idx} for prediction category {pred_label_idx}.")
    except Exception as e:
        print(f"Error occurred while generating the Waterfall Plot: {e}")
        import traceback
        traceback.print_exc()

    # e. SHAP Force Plot (Single Sample, Single Class) - Fitted to Binary Classification
    print("Generate SHAP Force Plot (Sample Example)...")
    try:
        sample_idx = 1
        if test_labels_for_vis and len(test_labels_for_vis) > sample_idx:
            true_label_name = class_names[test_labels_for_vis[sample_idx]]
        else:
            true_label_name = "Unknown"
        sample_probs = model_predict_fn(X_test_shap_vis_cpu[sample_idx:sample_idx + 1])
        pred_label_idx = np.argmax(sample_probs[0])
        pred_label_name = class_names[pred_label_idx]

        if pred_label_idx < len(shap_values) and isinstance(shap_values[pred_label_idx], np.ndarray):
            shap_val_for_sample = shap_values[pred_label_idx][sample_idx]
            feature_values_for_sample = X_test_shap_vis_cpu[sample_idx].numpy()
            base_value = expected_value[pred_label_idx]

            explanation_force = shap.Explanation(
                values=shap_val_for_sample,
                base_values=base_value,
                data=feature_values_for_sample,
                feature_names=feature_names
            )
            shap_html = shap.force_plot(base_value, shap_val_for_sample, feature_values_for_sample,
                                        feature_names=feature_names, show=False, matplotlib=False)
            shap.save_html(f"shap_force_sample_{sample_idx}_pred_{pred_label_name}.html", shap_html)
            print(
                f"  - Force Plot HTML for saved sample {sample_idx}: shap_force_sample_{sample_idx}_pred_{pred_label_name}.html")
        else:
            print(f"  - Error: Unable to obtain SHAP values for sample {sample_idx} for prediction category {pred_label_idx}.")
    except Exception as e:
        print(f"Error generating Force Plot: {e}")
        import traceback
        traceback.print_exc()

    # f. Error Case Analysis
    print("Analyze misclassified samples...")
    try:
        error_indices = np.where(np.array(test_labels_for_vis) != np.array(test_pred_for_vis))[0]
        print(f"A total of {len(error_indices)} misclassified samples were found.")

        for error_idx in error_indices[:3]:
            sample_idx = error_idx
            true_label = test_labels_for_vis[error_idx]
            pred_label = test_pred_for_vis[error_idx]
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]

            sample_probs = model_predict_fn(X_test_shap_vis_cpu[sample_idx:sample_idx + 1])
            pred_idx = np.argmax(sample_probs[0])
            if pred_idx < len(shap_values) and isinstance(shap_values[pred_idx], np.ndarray):
                shap_val_for_sample = shap_values[pred_idx][sample_idx]
                feature_values_for_sample = X_test_shap_vis_cpu[sample_idx].numpy()
                base_value = expected_value[pred_idx]

                explanation_error = shap.Explanation(
                    values=shap_val_for_sample,
                    base_values=base_value,
                    data=feature_values_for_sample,
                    feature_names=feature_names
                )

                plt.figure(figsize=(10, 8))
                shap.waterfall_plot(explanation_error, show=False, max_display=15)

                plt.title(f"Error Sample {sample_idx} | True={true_name}, Pred={pred_name}")
                plt.tight_layout()
                plt.savefig(
                    f'shap_waterfall_error_sample_{sample_idx}_t_{true_name}_p_{pred_name}.png',
                    dpi=150, bbox_inches='tight')
                plt.show()
                print(f"  - Waterfall Plot for saved error sample {sample_idx}.")
            else:
                print(f"  - Error: Unable to obtain SHAP values for sample {sample_idx} for prediction category {pred_idx}.")
    except Exception as e:
        print(f"Error occurred while analyzing error cases: {e}")
        import traceback
        traceback.print_exc()

    # --- 7. Restore Model Equipment ---
    if original_device.type != 'cpu':
        print(f"The model is being moved back to the original equipment. {original_device}...")
        model = model.to(original_device)
        print(f"The model has been returned to the device.: {next(model.parameters()).device}")

    print(f"{'=' * 50}")
    print("SHAP interpretability analysis completed.")
    print(f"{'=' * 50}")

try:
    df_temp = pd.read_csv(train_csv)
    feature_column_names = [col for col in df_temp.columns if col != 'delivery_risk']
    print(f"Read {len(feature_column_names)} feature names from the training data.")
except Exception as e:
    print(f"Unable to read feature names from training CSV: {e}")
    feature_column_names = None

add_shap_analysis(
    model=model,
    test_loader=test_loader,
    train_loader=train_loader,
    device=device,
    class_names=class_names,
    feature_names=feature_column_names,
    max_eval_samples=500,
    max_vis_samples=400
)