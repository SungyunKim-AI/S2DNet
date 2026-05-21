"""
One-vs-Rest 단일 모델 학습 및 단일 모델 추론.

1. Normal vs Others 학습
   python finetuning_ovr.py --cuda_visible_devices 0 --binary_type nl_vs_others

2. Neuropathy vs Others 학습
   python finetuning_ovr.py --cuda_visible_devices 1 --binary_type n_vs_others

3. Myopathy vs Others 학습
   python finetuning_ovr.py --cuda_visible_devices 2 --binary_type m_vs_others
"""

import os
import argparse
import yaml
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from finetuning import get_model, training, evaluate
from utils.native_scaler import NativeScalerWithGradNormCount as NativeScaler
from utils.datasets import FinetuneDataset_ovr
from utils.optim_factory import LayerDecayValueAssigner, create_optimizer
from utils.logger import save_report
from utils.native_scaler import cosine_scheduler
from utils.get_loss import get_loss_fn

def get_args():
    # Main parser
    parser = argparse.ArgumentParser('OvR single model fine-tuning script', add_help=False)
    
    # ===== Data and Dataset Parameters =====
    parser.add_argument('--cuda_visible_devices', type=str, default="0", help='visible devices')
    parser.add_argument('--output_dir', type=str, default='outputs/muscle-level', help='output directory')
    parser.add_argument('--inference', action='store_true', default=False, help='perform inference only')
    parser.add_argument('--meta_file_path_train', type=str, 
                        default='../data/preprocessed/labeled_seg_400_meta_data_train.parquet')
    parser.add_argument('--meta_file_path_valid', type=str, 
                        default='../data/preprocessed/labeled_seg_400_meta_data_valid.parquet')
    parser.add_argument('--pretrained_config_path', type=str, 
                        default='./outputs/pretrain/BimodalMAE_backbone/config.yaml')
    parser.add_argument('--pretrained_weights_path', type=str, default=None)
    parser.add_argument('--bag_size', default=30, type=int,
                        help='bag size for MIL')
    parser.add_argument('--denorm', default=True, type=bool,
                        help='denormalize input data')
    parser.add_argument('--binary_type', default='nl_vs_others', type=str, 
                        choices=['nl_vs_others', 'n_vs_others', 'm_vs_others'],
                        help='binary classification type: nl_vs_others (nl vs n+m), n_vs_others (n vs nl+m), m_vs_others (m vs nl+n)')
    
    # Architecture Parameters
    parser.add_argument('--num_heads_clf', default=4, type=int, help='number of attention heads for classifier')
    parser.add_argument('--attn_drop_rate_clf', default=0.0, type=float, help='attention dropout rate for classifier')
    
    # Training Parameters
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs') 
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int, help='gradient accumulation steps')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--layer_decay', default=0.75, type=float, help='layer decay rate')
    
    # Optimizer Parameters
    parser.add_argument('--opt', default='adamw', type=str, help='optimizer type')
    parser.add_argument('--opt_eps', default=1e-8, type=float, help='optimizer epsilon')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', help='optimizer betas')
    parser.add_argument('--clip_grad', type=float, default=None, help='gradient clipping norm')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--weight_decay_end', type=float, default=None, help='final weight decay value') 
    
    # Learning Rate Schedule Parameters
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--min_lr', type=float, default=0.0, metavar='LR', help='minimum learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='warmup epochs')
    
    # Loss Function Parameters
    parser.add_argument('--criterion', default='CrossEntropyLoss', type=str, help='loss function type')
    parser.add_argument('--focal_alpha', default=[0.5, 0.5], type=float, nargs='+', help='focal loss alpha values')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='focal loss gamma value')
    
    # ===== Single-model inference =====
    parser.add_argument('--model_nl_vs_others_path', type=str, default=None,
                        help='path to normal vs others checkpoint (when --binary_type nl_vs_others --inference)')
    parser.add_argument('--model_n_vs_others_path', type=str, default=None,
                        help='path to neuropathy vs others checkpoint (when --binary_type n_vs_others --inference)')
    parser.add_argument('--model_m_vs_others_path', type=str, default=None,
                        help='path to myopathy vs others checkpoint (when --binary_type m_vs_others --inference)')

    config_parser = argparse.ArgumentParser(description='Finetuning Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='./configs/finetune_ovr.yaml', type=str, 
                                metavar='FILE', help='YAML config file specifying default arguments')
    
    args_config, remaining = config_parser.parse_known_args()
    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)

    with open(args.pretrained_config_path, 'r') as f:
        pretrained_cfg = yaml.safe_load(f)
        # 기존 args에 pretrained config 병합 (기존 값이 우선)
        for k, v in pretrained_cfg.items():
            if not hasattr(args, k):  # 기존에 없는 키만 추가
                setattr(args, k, v)
    return args


def main(args):
    # Dataloaders
    train_dataset = FinetuneDataset_ovr(args.meta_file_path_train, args.bag_size, args.denorm, args.binary_type)
    val_dataset = FinetuneDataset_ovr(args.meta_file_path_valid, args.bag_size, args.denorm, args.binary_type)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True, drop_last=False
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.inference:
        model_path = getattr(args, f"model_{args.binary_type}_path")
        if model_path is None:
            raise ValueError(f"Inference requires --model_{args.binary_type}_path")
        checkpoint = torch.load(
            model_path, weights_only=False, map_location=device
        )['model_state_dict']
        model = get_model(args)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        val_stats = evaluate(model, val_loader, None, device)
        save_report(args.output_dir, val_stats)
        return
    
    # Model
    model = get_model(args)
    model.to(device)

    # Optimizer
    num_layers = model.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        print("Assigned values = %s" % str(assigner.values))
    else:
        assigner = None

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)
    
    optimizer = create_optimizer(
        args, model, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    # Learning rate scheduler
    num_training_steps_per_epoch = len(train_loader)
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, 
        args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs
    )

    # Weight decay scheduler
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # Loss function
    # if args.binary_type == 'nl_vs_others':
    #     args.focal_alpha = [0.6, 0.5]
    # elif args.binary_type == 'n_vs_others':
    #     args.focal_alpha = [0.42, 0.58]
    if args.binary_type =='m_vs_others':
        args.criterion = 'FocalLoss'
        args.focal_alpha = [0.03, 0.97]
        args.focal_gamma = 3
    # else:
    #     args.focal_alpha = None

    criterion = get_loss_fn(
        args.criterion,
        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma, 
        device=device
    )
    print("criterion = %s" % str(criterion))
    
    training(
        model, criterion, train_loader, val_loader, optimizer, 
        device, args, loss_scaler, lr_schedule_values, wd_schedule_values
    )

    args_dict = vars(args)
    yaml_path = Path(args.output_dir) / "config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(args_dict, f, allow_unicode=True, default_flow_style=False)


if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    output_dir = Path(args.output_dir) / args.binary_type
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    ckpts_dir = Path(args.output_dir) / 'ckpts'
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    # fix the seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = True

    main(args)
