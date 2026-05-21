"""
Ensemble inference using 3 trained One-vs-Rest individual models.
"""

import os
import argparse
import copy
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from finetuning import get_model
from utils.datasets import FinetuneDataset
from utils.get_loss import evaluate_metrics
from utils.logger import save_report


def get_args():
    parser = argparse.ArgumentParser('OvR ensemble inference script', add_help=False)
    parser.add_argument('--cuda_visible_devices', type=str, default='0', help='visible devices')
    parser.add_argument('--output_dir', type=str, default='outputs/muscle_level/ensemble',
                        help='output directory for inference results')
    parser.add_argument('--meta_file_path_valid', type=str,
                        default='../data/preprocessed/labeled_seg_400_meta_data_valid.parquet')
    parser.add_argument('--bag_size', default=50, type=int, help='bag size for MIL')
    parser.add_argument('--denorm', default=False, type=bool, help='denormalize input data')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')

    config_parser = argparse.ArgumentParser(description='Ensemble Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='./configs/finetune_ensemble.yaml', type=str,
                               metavar='FILE', help='YAML config for model defaults')
    args_config, remaining = config_parser.parse_known_args()
    if os.path.isfile(args_config.config):
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    
    required_paths = ['model_nl_vs_others_path', 'model_n_vs_others_path', 'model_m_vs_others_path']
    for path_attr in required_paths:
        if not hasattr(args, path_attr):
            raise ValueError(f"Missing required argument: '{path_attr}'")

    return args


def load_ovr_model(ckpt_path, device):
    config_path = Path(ckpt_path).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        args_ovr = argparse.Namespace(**config)
    args_ovr.inference = True
    args_ovr.num_classes = 2 

    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)['model_state_dict']
    model = get_model(args_ovr)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def ensemble_inference(args, dataloader, device, output_dir):
    """Ensemble 3 one-vs-others models to produce final 3-class prediction"""
    model_nl = load_ovr_model(args.model_nl_vs_others_path, device)
    model_n = load_ovr_model(args.model_n_vs_others_path, device)
    model_m = load_ovr_model(args.model_m_vs_others_path, device)

    all_probabilities, all_labels = [], []
    all_group_keys = []
    rows_for_csv = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for (inputs, targets, norm_dict, group_key) in dataloader:
        inputs = {
            'time': inputs['time'].to(device, non_blocking=True),
            'freq': inputs['freq'].to(device, non_blocking=True),
        }
        target = targets.to(device, non_blocking=True)
        norm_dict = {
            'mean': norm_dict['mean'].to(device, non_blocking=True),
            'std': norm_dict['std'].to(device, non_blocking=True),
        }

        with torch.amp.autocast("cuda"):
            output_nl = model_nl(inputs, norm_dict)['cls']
            output_n = model_n(inputs, norm_dict)['cls']
            output_m = model_m(inputs, norm_dict)['cls']

            proba_nl = F.softmax(output_nl, dim=1)
            proba_n = F.softmax(output_n, dim=1)
            proba_m = F.softmax(output_m, dim=1)

            batch_size = target.size(0)
            final_proba = torch.zeros(batch_size, 3).to(device)
            for i in range(batch_size):
                p0 = proba_nl[i, 1]
                p1 = proba_n[i, 1]
                p2 = proba_m[i, 1]

                total_p = p0 + p1 + p2
                final_proba[i, 0] = p0 / total_p
                final_proba[i, 1] = p1 / total_p
                final_proba[i, 2] = p2 / total_p

            loss = criterion(final_proba, target)

        total_loss += loss.item()
        # Collect output_nl[:,1], output_n[:,1], output_m[:,1], target per sample (for CSV saving)
        batch_size = target.size(0)
        onl1 = output_nl[:, 1].cpu().float().numpy()
        on1 = output_n[:, 1].cpu().float().numpy()
        om1 = output_m[:, 1].cpu().float().numpy()
        tgt = target.cpu().numpy()
        for i in range(batch_size):
            rows_for_csv.append({
                'output_nl_cls1': float(onl1[i]),
                'output_n_cls1': float(on1[i]),
                'output_m_cls1': float(om1[i]),
                'target': int(tgt[i]),
            })

        for i in range(batch_size):
            gk_0 = group_key[0][i] if torch.is_tensor(group_key[0]) else group_key[0][i] if hasattr(group_key[0], '__getitem__') else group_key[0]
            gk_1 = group_key[1][i] if isinstance(group_key[1], (list, tuple)) else group_key[1]
            gk_2 = group_key[2][i] if torch.is_tensor(group_key[2]) else group_key[2][i] if hasattr(group_key[2], '__getitem__') else group_key[2]
            all_group_keys.append({
                'pid': int(gk_0.item()) if torch.is_tensor(gk_0) else int(gk_0),
                'visit_date': gk_1 if isinstance(gk_1, str) else (gk_1[0] if (hasattr(gk_1, '__len__') and len(gk_1) > 0) else gk_1),
                'muscle_index': int(gk_2.item()) if torch.is_tensor(gk_2) else int(gk_2),
                'label': int(target[i].item()),
                'prob_nl': float(final_proba[i, 0].item()),
                'prob_n': float(final_proba[i, 1].item()),
                'prob_m': float(final_proba[i, 2].item()),
            })

        all_labels.extend(target.cpu().numpy())
        all_probabilities.extend(final_proba.cpu().numpy())

    all_probabilities = np.array(all_probabilities)
    val_stats = evaluate_metrics(all_labels, all_probabilities, classification_rep=True)
    val_stats['loss'] = total_loss / len(dataloader)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df_csv = pd.DataFrame(rows_for_csv)
    csv_path = os.path.join(output_dir, "ensemble_outputs_and_targets.csv")
    df_csv.to_csv(csv_path, index=False)
    print(f"Per-sample outputs and targets saved to {csv_path} ({len(df_csv)} rows)")

    # Save labels+probabilities → enables visualization (e.g., ROC curve) later without re-running the model
    df_keys = pd.DataFrame(all_group_keys)
    roc_data_path = os.path.join(output_dir, "ensemble_labels_and_proba.parquet")
    df_keys.to_parquet(roc_data_path, index=False)
    print(f"Labels and probabilities saved to {roc_data_path} ({len(df_keys)} rows, for ROC/analysis)")
    return val_stats


def main():
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    val_dataset = FinetuneDataset(args.meta_file_path_valid, args.bag_size, args.denorm)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True, drop_last=False
    )

    val_stats = ensemble_inference(args, val_loader, device, args.output_dir)
    save_report(args.output_dir, val_stats)
    print("Ensemble inference done. Results saved to", args.output_dir)


if __name__ == '__main__':
    main()
