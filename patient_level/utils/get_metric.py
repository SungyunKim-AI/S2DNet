import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
)


def _fmt_ci(value, ci_tuple):
    """Format metric with 95% CI: 0.8234 (0.7891 - 0.8576)"""
    if ci_tuple is None or (isinstance(ci_tuple, tuple) and any(v != v for v in ci_tuple)):
        return f"{value:.4f}"
    lo, hi = ci_tuple
    return f"{value:.4f} ({lo:.4f} - {hi:.4f})"


def _bootstrap_ci(y_true, y_pred_proba, num_classes, n_bootstrap=1000, seed=42):
    """Bootstrap 95% CI for AUROC, AUPRC, F1 (macro/micro/weighted + per-class)."""
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    n = len(y_true)

    slots = {
        'auroc_macro': [], 'auroc_weighted': [],
        'auprc_macro': [], 'auprc_weighted': [],
        'f1_macro': [], 'f1_micro': [], 'f1_weighted': [],
    }
    auroc_per = [[] for _ in range(num_classes)]
    auprc_per = [[] for _ in range(num_classes)]
    f1_per    = [[] for _ in range(num_classes)]

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred_proba[idx]
        if len(np.unique(yt)) < num_classes:
            continue
        try:
            slots['auroc_macro'].append(roc_auc_score(yt, yp, multi_class='ovr', average='macro'))
            slots['auroc_weighted'].append(roc_auc_score(yt, yp, multi_class='ovr', average='weighted'))
            per_a = [roc_auc_score((yt == i).astype(int), yp[:, i]) for i in range(num_classes)]
            for i in range(num_classes):
                auroc_per[i].append(per_a[i])

            per_p = [average_precision_score((yt == i).astype(int), yp[:, i]) for i in range(num_classes)]
            slots['auprc_macro'].append(float(np.mean(per_p)))
            _, cnts = np.unique(yt, return_counts=True)
            w = cnts / cnts.sum() if len(cnts) == num_classes else np.ones(num_classes) / num_classes
            slots['auprc_weighted'].append(float(np.dot(per_p, w)))
            for i in range(num_classes):
                auprc_per[i].append(per_p[i])

            ypred = np.argmax(yp, axis=1)
            slots['f1_macro'].append(f1_score(yt, ypred, average='macro', zero_division=0))
            slots['f1_micro'].append(f1_score(yt, ypred, average='micro', zero_division=0))
            slots['f1_weighted'].append(f1_score(yt, ypred, average='weighted', zero_division=0))
            pf1 = f1_score(yt, ypred, average=None, zero_division=0)
            for i in range(num_classes):
                f1_per[i].append(pf1[i] if i < len(pf1) else 0.0)
        except Exception:
            continue

    def _ci(vals):
        if len(vals) == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    return {
        'auroc_macro_ci':    _ci(slots['auroc_macro']),
        'auroc_weighted_ci': _ci(slots['auroc_weighted']),
        'auroc_per_ci':      [_ci(auroc_per[i]) for i in range(num_classes)],
        'auprc_macro_ci':    _ci(slots['auprc_macro']),
        'auprc_weighted_ci': _ci(slots['auprc_weighted']),
        'auprc_per_ci':      [_ci(auprc_per[i]) for i in range(num_classes)],
        'f1_macro_ci':       _ci(slots['f1_macro']),
        'f1_micro_ci':       _ci(slots['f1_micro']),
        'f1_weighted_ci':    _ci(slots['f1_weighted']),
        'f1_per_ci':         [_ci(f1_per[i]) for i in range(num_classes)],
    }

def calculate_auroc(y_true, y_pred_proba, num_classes=4):
    """AUROC 계산 (Macro, Weighted, Per-class)"""
    macro_auroc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    weighted_auroc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    
    # 각 클래스별 AUROC 계산
    per_class_auroc = []
    for i in range(num_classes):
        try:
            class_auroc = roc_auc_score((y_true == i).astype(int), y_pred_proba[:, i])
            per_class_auroc.append(class_auroc)
        except ValueError:
            # 해당 클래스가 validation set에 없는 경우
            per_class_auroc.append(0.0)
    return macro_auroc, weighted_auroc, per_class_auroc


def calculate_auprc(y_true, y_pred_proba, num_classes=4):
    """AUPRC 계산 (Macro, Weighted, Per-class). One-vs-rest 기준."""
    # 각 클래스별 AUPRC 계산
    per_class_auprc = []
    for i in range(num_classes):
        try:
            class_auprc = average_precision_score((y_true == i).astype(int), y_pred_proba[:, i])
            per_class_auprc.append(class_auprc)
        except ValueError:
            per_class_auprc.append(0.0)
    per_class_auprc = np.array(per_class_auprc)
    macro_auprc = float(np.mean(per_class_auprc))
    # Weighted: 클래스별 샘플 수 비율로 가중 평균
    _, counts = np.unique(y_true, return_counts=True)
    if len(counts) == num_classes:
        weights = counts / counts.sum()
        weighted_auprc = float(np.sum(per_class_auprc * weights))
    else:
        weighted_auprc = macro_auprc
    return macro_auprc, weighted_auprc, per_class_auprc.tolist()



def save_metrics(y_true, y_pred, y_pred_proba, class_names, save_path, epoch,
                 macro_auroc, weighted_auroc, per_class_auroc,
                 macro_auprc=None, weighted_auprc=None, per_class_auprc=None,
                 n_bootstrap=1000):
    """메트릭을 텍스트 파일로 저장 (AUROC, AUPRC, F1 + Bootstrap 95% CI 포함)"""
    num_classes = len(class_names)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # F1 Score (macro, micro, weighted, per-class)
    f1_macro    = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    f1_micro    = f1_score(y_true, y_pred, average='micro',    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_per      = f1_score(y_true, y_pred, average=None,       zero_division=0).tolist()

    # Classification Report & Confusion Matrix
    class_report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    if per_class_auprc is None:
        per_class_auprc = [0.0] * num_classes
    if macro_auprc is None:
        macro_auprc = float(np.mean(per_class_auprc))
    if weighted_auprc is None:
        weighted_auprc = macro_auprc

    # Bootstrap 95% CI
    print(f"  Computing bootstrap 95% CI (n={n_bootstrap})...")
    ci = _bootstrap_ci(y_true, y_pred_proba, num_classes, n_bootstrap=n_bootstrap)

    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Best Model Metrics (Epoch {epoch})\n")
        f.write("=" * 80 + "\n\n")

        # Overall Metrics
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:            {accuracy:.4f}\n")
        f.write(f"F1 (Macro):          {_fmt_ci(f1_macro,    ci['f1_macro_ci'])}\n")
        f.write(f"F1 (Micro):          {_fmt_ci(f1_micro,    ci['f1_micro_ci'])}\n")
        f.write(f"F1 (Weighted):       {_fmt_ci(f1_weighted, ci['f1_weighted_ci'])}\n")
        f.write(f"AUROC (Macro):       {_fmt_ci(macro_auroc,    ci['auroc_macro_ci'])}\n")
        f.write(f"AUROC (Weighted):    {_fmt_ci(weighted_auroc, ci['auroc_weighted_ci'])}\n")
        f.write(f"AUPRC (Macro):       {_fmt_ci(macro_auprc,    ci['auprc_macro_ci'])}\n")
        f.write(f"AUPRC (Weighted):    {_fmt_ci(weighted_auprc, ci['auprc_weighted_ci'])}\n\n")

        # Per-class F1
        f.write("PER-CLASS F1\n")
        f.write("-" * 80 + "\n")
        for i, class_name in enumerate(class_names):
            val = f1_per[i] if i < len(f1_per) else 0.0
            f.write(f"  Class {i} ({class_name}): {_fmt_ci(val, ci['f1_per_ci'][i])}\n")
        f.write("\n")

        # Per-class AUROC
        f.write("PER-CLASS AUROC\n")
        f.write("-" * 80 + "\n")
        for i, (class_name, auroc) in enumerate(zip(class_names, per_class_auroc)):
            f.write(f"  Class {i} ({class_name}): {_fmt_ci(auroc, ci['auroc_per_ci'][i])}\n")
        f.write("\n")

        # Per-class AUPRC
        f.write("PER-CLASS AUPRC\n")
        f.write("-" * 80 + "\n")
        for i, (class_name, auprc) in enumerate(zip(class_names, per_class_auprc)):
            f.write(f"  Class {i} ({class_name}): {_fmt_ci(auprc, ci['auprc_per_ci'][i])}\n")
        f.write("\n")

        # Classification Report
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 80 + "\n")
        f.write(class_report)
        f.write("\n")

        # Confusion Matrix
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 80 + "\n")
        header_label = 'True\\Pred'
        f.write(f"{header_label:<20}")
        for name in class_names:
            f.write(f"{name[:15]:>15}")
        f.write("\n")
        for i, row in enumerate(cm):
            f.write(f"{class_names[i][:20]:<20}")
            for val in row:
                f.write(f"{val:>15}")
            f.write("\n")
        f.write("\n")

        # Confusion Matrix (Normalized)
        f.write("CONFUSION MATRIX (Normalized)\n")
        f.write("-" * 80 + "\n")
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        f.write(f"{header_label:<20}")
        for name in class_names:
            f.write(f"{name[:15]:>15}")
        f.write("\n")
        for i, row in enumerate(cm_normalized):
            f.write(f"{class_names[i][:20]:<20}")
            for val in row:
                f.write(f"{val:>15.4f}")
            f.write("\n")
        f.write("\n")

        f.write("=" * 80 + "\n")

