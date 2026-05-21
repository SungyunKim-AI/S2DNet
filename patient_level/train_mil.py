'''
python train_mil.py \
--output_dir outputs/BimodalMAE --gpu 2 \
--muscle_dim 192 --hidden_dim 128 --num_heads 2 --num_layers 3 \
--train_data ./data/BimodalMAE_train_embed.parquet \
--valid_data ./data/BimodalMAE_valid_embed.parquet
'''

import os
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from dataset import MILDataset
from models import MultiExpertGatedMIL
from utils.get_metric import calculate_auroc, calculate_auprc, save_metrics
from utils.get_loss import FocalLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train Gated Attention MIL Model')
    
    # Data paths
    parser.add_argument('--train_data', type=str, default='/home/coder/workspace/data/BMI_nEMG/3_patient-level_clf/data/BimodalMAE_train_embed.parquet',
                        help='Training data path (parquet file)')
    parser.add_argument('--valid_data', type=str, default='/home/coder/workspace/data/BMI_nEMG/3_patient-level_clf/data/BimodalMAE_valid_embed.parquet',
                        help='Validation data path (parquet file)')
    
    # Model configuration
    parser.add_argument('--muscle_dim', type=int, default=192,
                        help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (for transformer model)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of transformer layers (for transformer model)')
    parser.add_argument('--ablation_mode', type=str, default='full',
                        help='Ablation mode: full, no_muscle, no_anatomy')
    
    # Training configuration (when bag_size is fixed, sample shape is consistent so batch_size can be increased)
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (can be raised to 8-32 when bag size is fixed)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch_size * accum_steps)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--focal_gamma', type=float, default=2.0, 
                        help='Focal loss gamma')
    
    # Early stopping & Scheduler
    parser.add_argument('--patience', type=int, default=25,
                        help='Early stopping patience')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Learning rate scheduler factor')
    parser.add_argument('--scheduler_patience', type=int, default=3,
                        help='Learning rate scheduler patience')
    
    parser.add_argument('--output_dir', type=str, default='outputs/gated_attention_mil_v1',
                        help='Path to save output directory')
    parser.add_argument('--gpu', type=str, default='1',
                        help='CUDA visible devices')
    parser.add_argument('--bag_size', type=int, default=10,
                        help='Fixed bag size (0 = variable). With 10: zero-pad if fewer, randomly sample 10 if more, with padding mask')
    
    return parser.parse_args()


def train_one_epoch(model, train_loader, criterion, optimizer, device, accum_steps, grad_clip, use_padding_mask=False):
    """Run training for one epoch. If use_padding_mask=True, batches are (inputs, labels, padding_mask)."""
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(train_loader):
        if use_padding_mask:
            inputs, labels, padding_mask = batch
            padding_mask = padding_mask.to(device, non_blocking=True)
        else:
            inputs, labels = batch
            padding_mask = None
        # When using fixed bag, pass (B, bag_size, D) as-is; for variable bag (1, N, D) → squeeze(0)
        if not use_padding_mask:
            inputs = inputs.squeeze(0).to(device, non_blocking=True)
        else:
            inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if padding_mask is not None:
            logits, _ = model(inputs, padding_mask=padding_mask)
        else:
            logits, _ = model(inputs)
        loss = criterion(logits, labels)
        
        # Gradient Accumulation
        loss = loss / accum_steps
        loss.backward()
        
        if (i + 1) % accum_steps == 0:
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss += loss.item() * accum_steps
        _, predicted = torch.max(logits, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_acc = 100 * train_correct / train_total
    avg_loss = train_loss / len(train_loader)
    
    return avg_loss, train_acc


def evaluate(model, valid_loader, criterion, device, num_classes, use_padding_mask=False):
    """Run validation and return metrics. If use_padding_mask=True, batches are (inputs, labels, padding_mask)."""
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_pred_proba = []
    all_pred = []
    all_labels = []
    
    with torch.no_grad():
        for batch in valid_loader:
            if use_padding_mask:
                inputs, labels, padding_mask = batch
                padding_mask = padding_mask.to(device, non_blocking=True)
            else:
                inputs, labels = batch
                padding_mask = None
            if not use_padding_mask:
                inputs = inputs.squeeze(0).to(device, non_blocking=True)
            else:
                inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if padding_mask is not None:
                logits, _ = model(inputs, padding_mask=padding_mask)
            else:
                logits, _ = model(inputs)
            loss = criterion(logits, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            # Collect probabilities and labels for AUROC computation
            probs = torch.softmax(logits, dim=1)
            all_pred_proba.append(probs.cpu().numpy())
            all_pred.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Compute metrics
    all_pred_proba = np.concatenate(all_pred_proba, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    val_acc = 100 * val_correct / val_total
    val_f1_macro = f1_score(all_labels, all_pred, average='macro', zero_division=0)
    avg_loss = val_loss / len(valid_loader)
    val_auroc_macro, val_auroc_weighted, per_class_auroc = calculate_auroc(all_labels, all_pred_proba, num_classes=num_classes)
    val_auprc_macro, val_auprc_weighted, per_class_auprc = calculate_auprc(all_labels, all_pred_proba, num_classes=num_classes)
    
    return {
        'loss': avg_loss,
        'accuracy': val_acc,
        'f1_macro': val_f1_macro,
        'auroc_macro': val_auroc_macro,
        'auroc_weighted': val_auroc_weighted,
        'per_class_auroc': per_class_auroc,
        'auprc_macro': val_auprc_macro,
        'auprc_weighted': val_auprc_weighted,
        'per_class_auprc': per_class_auprc,
        'predictions': all_pred,
        'probabilities': all_pred_proba,
        'labels': all_labels
    }


def train_model(model, train_loader, valid_loader, num_epochs=20, learning_rate=1e-4, accum_steps=4,
                grad_clip=1.0, patience=5, focal_gamma=3.0, model_save_path='best_mil_model.pth', weight_decay=1e-5,
                scheduler_factor=0.5, scheduler_patience=3, metrics_save_path='best_model_metrics.txt',
                class_names=None, use_padding_mask=False):
    """Main model training function. use_padding_mask: set True when batches are (inputs, labels, padding_mask)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss & Optimizer
    # focal_alpha = [1.0, 1.0, 5.0, 10.0]
    counts = np.array([100, 100, 10, 5], dtype=np.float32)
    weights = 1.0 / counts
    weights = weights / weights.sum() * 4
    focal_alpha = weights.tolist()
    print("focal_alpha:", focal_alpha)
    # exit()

    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=scheduler_factor, patience=scheduler_patience
    )
    
    # Variables for early stopping
    best_auroc = 0.0
    patience_counter = 0
    num_classes = len(class_names)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, accum_steps, grad_clip,
            use_padding_mask=use_padding_mask,
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validation
        val_metrics = evaluate(model, valid_loader, criterion, device, num_classes, use_padding_mask=use_padding_mask)
        print(f"Valid Loss: {val_metrics['loss']:.4f}, Valid F1: {val_metrics['f1_macro']:.4f}, "
              f"Valid AUROC: {val_metrics['auroc_macro']:.4f}, "
              f"Valid AUPRC: {val_metrics['auprc_macro']:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(val_metrics['auroc_macro'])
        
        # Save best model (based on AUROC)
        if val_metrics['auroc_macro'] > best_auroc:
            best_auroc = val_metrics['auroc_macro']
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            
            # Save metrics (including AUROC, AUPRC)
            save_metrics(
                val_metrics['labels'],
                val_metrics['predictions'],
                val_metrics['probabilities'],
                class_names[:num_classes],
                metrics_save_path,
                epoch+1,
                val_metrics['auroc_macro'],
                val_metrics['auroc_weighted'],
                val_metrics['per_class_auroc'],
                macro_auprc=val_metrics['auprc_macro'],
                weighted_auprc=val_metrics['auprc_weighted'],
                per_class_auprc=val_metrics['per_class_auprc'],
            )
            
            print(f"--> Best model saved! (AUROC: {best_auroc:.4f}, AUPRC: {val_metrics['auprc_macro']:.4f}, F1: {val_metrics['f1_macro']:.4f})")
            print(f"--> Metrics saved to: {metrics_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered! Best AUROC: {best_auroc:.4f}")
                break



if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.benchmark = True

    args = parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = vars(args)
    config_save_path = str(args.output_dir / 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Dataset & DataLoader (if bag_size>0: fixed bag + zero-padding/random sampling + padding_mask)
    train_df = pd.read_parquet(args.train_data)
    valid_df = pd.read_parquet(args.valid_data)
    if getattr(args, 'bag_size', 0) > 0:
        train_dataset = MILDataset(train_df, bag_size=args.bag_size, seed=42)
        valid_dataset = MILDataset(valid_df, bag_size=args.bag_size, seed=43)
        use_padding_mask = True
    else:
        raise ValueError("Need bag_size")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    model = MultiExpertGatedMIL(
        muscle_dim=args.muscle_dim,
        anatomy_dim=64,
        hidden_dim=args.hidden_dim, 
        num_classes=args.num_classes, 
        num_heads=args.num_heads, 
        num_layers=args.num_layers,
        ablation_mode=args.ablation_mode
    )

    # Define class names
    if args.num_classes == 4:
        class_names = ['normal', 'radiculopathy', 'focal neuropathy', 'polyneuropathy']
    elif args.num_classes == 5:
        class_names = ['normal', 'radiculopathy', 'focal neuropathy', 'polyneuropathy', 'systemic myopathy']
    else:
        raise ValueError(f"Invalid number of classes: {args.num_classes}")
    
    # Set model and metrics save paths
    model_save_path = str(args.output_dir / 'best_model.pth')
    metrics_save_path = str(args.output_dir / 'best_model_metrics.txt')
    
    # Train Model
    train_model(
        model,
        train_loader,
        valid_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        accum_steps=args.accum_steps,
        grad_clip=args.grad_clip,
        patience=args.patience,
        focal_gamma=args.focal_gamma,
        model_save_path=model_save_path,
        weight_decay=args.weight_decay,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        metrics_save_path=metrics_save_path,
        class_names=class_names,
        use_padding_mask=use_padding_mask
    )
