import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, classification_report, confusion_matrix
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MaskedMSELoss(nn.Module):
    """MSE loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param norm_pix: Normalized pixel loss
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix=False, reduction='mean'):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.norm_pix = norm_pix
        self.reduction = reduction

    def patchify(self, imgs, nw):
        p = self.scale_factor
        x = rearrange(imgs, "b c (nw p) -> b nw (p c)", nw=nw, p=p)
        return x

    def unpatchify(self, x, nw):
        p = self.scale_factor
        imgs = rearrange(x, "b nw (p c) -> b c (nw p)", nw=nw, p=p)
        return imgs

    def forward(self, input, target, mask=None):

        W = list(input.shape[-1:])[0]
        nw = W // self.scale_factor

        if self.norm_pix:
            target = self.patchify(target, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nw)

        loss = F.mse_loss(input, target, reduction='none')

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            mask = F.interpolate(mask.unsqueeze(1).float(), size=W, mode='nearest').squeeze(1)
            loss = loss.mean(dim=1)  # B, C, W -> B, W
            loss = loss * mask

            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)

            if self.reduction == 'mean':
                loss = loss.nanmean()  # If this is ever nan, we want it to stop training
            elif self.reduction == 'sum':
                loss = loss.sum()
            else:
                loss = loss
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training

        return loss



class FocalLoss(torch.nn.Module):
    def __init__(self, alpha='None', gamma=3, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha != 'None':
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch, num_classes], targets: [batch] (class index)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_fn(criterion, weights=None, device='cpu', focal_alpha=None, focal_gamma=None):
    if criterion == 'CrossEntropyLoss':
        if weights is not None:
            return torch.nn.CrossEntropyLoss(weight=weights.to(device))
        else:
            return torch.nn.CrossEntropyLoss()
    
    elif criterion == 'FocalLoss':
        if focal_alpha is None or focal_gamma is None:
            raise ValueError("FocalLoss requires both focal_alpha and focal_gamma parameters")
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    else:
        raise ValueError(f"Not Implemented loss function: {criterion}")



class OvRLoss(nn.Module):
    """
    One-vs-Rest Loss using BCEWithLogitsLoss.
    Converts multi-class targets to one-hot encoding and calculates BCE loss.

    Class distribution & Weights:
        Class 'nl' (0): 1792 positive, 2785 negative -> pos_weight=1.5541
        Class 'n' (1): 2666 positive, 1911 negative -> pos_weight=0.7168
        Class 'm' (2): 119 positive, 4458 negative -> pos_weight=37.4622
    """
    def __init__(self, device, pos_weight=None):
        super(OvRLoss, self).__init__()
        self.device = device
        # pos_weight: Tensor of shape [num_classes]
        if pos_weight is not None:
            self.pos_weight = pos_weight.to(device)
        else:
            self.pos_weight = None
            
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits, targets):
        # logits: (Batch, Num_Classes)
        # targets: (Batch) -> need to be converted to one-hot
        
        num_classes = logits.shape[1]
        
        # Convert targets to one-hot encoding
        # targets shape: (Batch,) -> (Batch, Num_Classes)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float().to(self.device)
        
        # Calculate BCE loss
        loss = self.criterion(logits, targets_one_hot)
        return loss


def _bootstrap_ci(y_true, y_probs, n_bootstrap=1000, seed=42):
    """Bootstrap 95% CI for AUROC, AUPRC (macro + per-class), macro-F1, weighted-F1, per-class F1."""
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    n = len(y_true)
    n_classes = y_probs.shape[1]

    aurocs, auprcs, macro_f1s, weighted_f1s = [], [], [], []
    aurocs_per_class = [[] for _ in range(n_classes)]
    auprcs_per_class = [[] for _ in range(n_classes)]
    f1s_per_class = [[] for _ in range(n_classes)]

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_probs[idx]
        if len(np.unique(yt)) < n_classes:
            continue
        try:
            yp_ohe = label_binarize(yt, classes=range(n_classes))
            aurocs.append(roc_auc_score(yp_ohe, yp, multi_class='ovr', average='macro'))
            per_auroc = roc_auc_score(yp_ohe, yp, multi_class='ovr', average=None)
            for i in range(n_classes):
                aurocs_per_class[i].append(per_auroc[i])

            per_auprc = [average_precision_score(yp_ohe[:, i], yp[:, i]) for i in range(n_classes)]
            auprcs.append(np.mean(per_auprc))
            for i in range(n_classes):
                auprcs_per_class[i].append(per_auprc[i])

            ypred = np.argmax(yp, axis=1)
            macro_f1s.append(f1_score(yt, ypred, average='macro', zero_division=0))
            weighted_f1s.append(f1_score(yt, ypred, average='weighted', zero_division=0))
            per_f1 = f1_score(yt, ypred, average=None, zero_division=0)
            for i in range(n_classes):
                f1s_per_class[i].append(per_f1[i] if i < len(per_f1) else 0.0)
        except Exception:
            continue

    def _ci(vals):
        if len(vals) == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    return {
        'auroc_ci': _ci(aurocs),
        'auroc_ci_per_class': [_ci(aurocs_per_class[i]) for i in range(n_classes)],
        'auprc_ci': _ci(auprcs),
        'auprc_ci_per_class': [_ci(auprcs_per_class[i]) for i in range(n_classes)],
        'macro_f1_ci': _ci(macro_f1s),
        'weighted_f1_ci': _ci(weighted_f1s),
        'f1_ci_per_class': [_ci(f1s_per_class[i]) for i in range(n_classes)],
    }


def evaluate_metrics(y_true, y_probs, classification_rep=False, bootstrap=True, n_bootstrap=1000):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_preds = np.argmax(y_probs, axis=1)
    acc = accuracy_score(y_true, y_preds)
    f1_macro = f1_score(y_true, y_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_preds, average='weighted', zero_division=0)
    f1_per_class = f1_score(y_true, y_preds, average=None, zero_division=0).tolist()

    if y_probs.shape[1] == 2:
        roc_aucs = None
        auc = roc_auc_score(y_true, y_probs[:, 1])
        auprc_per_class = None
        auprc = average_precision_score(y_true, y_probs[:, 1], average='macro')
    else:
        y_true_one_hot = label_binarize(y_true, classes=range(y_probs.shape[1]))
        roc_aucs = roc_auc_score(y_true_one_hot, y_probs, multi_class='ovr', average=None)
        auc = np.mean(roc_aucs)
        auprc_per_class = [
            average_precision_score(y_true_one_hot[:, i], y_probs[:, i])
            for i in range(y_probs.shape[1])
        ]
        auprc = np.mean(auprc_per_class)
        print(f"AUC nl: {roc_aucs[0]:.4f} / n: {roc_aucs[1]:.4f} / m: {roc_aucs[2]:.4f}")
        print(f"AUPRC nl: {auprc_per_class[0]:.4f} / n: {auprc_per_class[1]:.4f} / m: {auprc_per_class[2]:.4f}")

    print(f'VAL: Acc {acc:.5f} Auc {auc:.5f} AUPRC {auprc:.5f} macro-F1 {f1_macro:.5f} weighted-F1 {f1_weighted:.5f}')

    val_stats = {
        'acc': acc, 'auc': auc, 'auprc': auprc,
        'roc_aucs': roc_aucs, 'auprc_per_class': auprc_per_class,
        'f1': f1_macro, 'f1_weighted': f1_weighted, 'f1_per_class': f1_per_class,
    }

    if bootstrap:
        print(f"Computing bootstrap 95% CI (n={n_bootstrap})...")
        ci = _bootstrap_ci(y_true, y_probs, n_bootstrap=n_bootstrap)
        val_stats.update(ci)

    if classification_rep:
        clf_report = classification_report(y_true, y_preds, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_preds)
        val_stats['classification_report'] = clf_report
        val_stats['confusion_matrix'] = conf_matrix

    return val_stats
