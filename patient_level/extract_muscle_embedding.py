import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pathlib import Path
from argparse import Namespace
from tqdm import tqdm
import pandas as pd
import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader
from utils.datasets import FinetuneDataset
from finetuning import get_model


def _load_config_from_ovr_folder(config_path):
    """Load config.yaml from OvR folder (nl_vs_others / n_vs_others / m_vs_others) and merge pretrained config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    args = Namespace(**cfg)
    args.inference = True
    args.num_classes = 2
    # Backbone config for get_model: merge pretrained config (same as finetuning.py)
    if getattr(args, "pretrained_config_path", None) and Path(args.pretrained_config_path).exists():
        with open(args.pretrained_config_path, "r") as f:
            pretrained_cfg = yaml.safe_load(f)
        for k, v in pretrained_cfg.items():
            if not hasattr(args, k):
                setattr(args, k, v)
    return args


def load_ovr_model(ckpt_path, device):
    """Load model using only checkpoint path, reading config.yaml from the grandparent folder (e.g., nl_vs_others)."""
    ckpt_path = Path(ckpt_path).resolve()
    config_path = ckpt_path.parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"OvR config not found: {config_path}")
    args = _load_config_from_ovr_folder(config_path)
    args.inference = True

    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)["model_state_dict"]
    model = get_model(args)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(model_nl, model_n, model_m, dataloader, device):

    all_embeds = []
    for (inputs, targets, norm_dict, group_key) in tqdm(dataloader, total=len(dataloader)):
        inputs = {
            'time': inputs['time'].to(device, non_blocking=True),
            'freq': inputs['freq'].to(device, non_blocking=True),
        }
        norm_dict = {
            'mean': norm_dict['mean'].to(device, non_blocking=True),
            'std': norm_dict['std'].to(device, non_blocking=True),
        }

        with torch.amp.autocast("cuda"):
            # Bag-level embedding from each OvR model (B, 1, dim_tokens)
            embed_nl = model_nl(inputs, norm_dict, return_embed=True)['cls']
            embed_n = model_n(inputs, norm_dict, return_embed=True)['cls']
            embed_m = model_m(inputs, norm_dict, return_embed=True)['cls']

        # (B, 1, dim) -> (B, dim) (dim == 192)
        embed_nl_np = embed_nl.squeeze(1).cpu().numpy()
        embed_n_np = embed_n.squeeze(1).cpu().numpy()
        embed_m_np = embed_m.squeeze(1).cpu().numpy()
        batch_size = embed_nl_np.shape[0]

        for i in range(batch_size):
            pid = group_key[0][i]
            pid = int(pid.item()) if torch.is_tensor(pid) else int(pid)
            visit_date = group_key[1][i] if isinstance(group_key[1], (list, tuple)) else group_key[1]
            muscle_index = group_key[2][i]
            muscle_index = int(muscle_index.item()) if torch.is_tensor(muscle_index) else int(muscle_index)
            all_embeds.append({
                'pid': pid,
                'visit_date': visit_date,
                'muscle_index': muscle_index,
                'embed_nl': embed_nl_np[i, :],
                'embed_n': embed_n_np[i, :],
                'embed_m': embed_m_np[i, :],
            })

    return all_embeds

if __name__ == "__main__":
    # Only model paths are specified; config.yaml is loaded from each path's grandparent folder (nl_vs_others / n_vs_others / m_vs_others)
    model_nl_vs_others_path = "./outputs/muscle_level/BimodalMAE_OvR/nl_vs_others/ckpts/BMIRC_022.pth"
    model_n_vs_others_path = "./outputs/muscle_level/BimodalMAE_OvR/n_vs_others/ckpts/BMIRC_017.pth"
    model_m_vs_others_path = "./outputs/muscle_level/BimodalMAE_OvR/m_vs_others/ckpts/BMIRC_022.pth"

    # model_nl_vs_others_path = "./outputs/muscle_level/BimodalMAE_OvR_tmp/nl_vs_others/ckpts/BMIRC_020.pth"
    # model_n_vs_others_path = "./outputs/muscle_level/BimodalMAE_OvR_tmp/n_vs_others/ckpts/BMIRC_008.pth"
    # model_m_vs_others_path = "./outputs/muscle_level/BimodalMAE_OvR_tmp/m_vs_others/ckpts/BMIRC_008.pth"

    # Data/runtime config is loaded from the OvR folder config of the first model path
    args = _load_config_from_ovr_folder(Path(model_nl_vs_others_path).resolve().parent.parent / "config.yaml")
    output_dir = "./outputs/muscle_level/BimodalMAE_OvR/ensemble"
    args.output_dir = output_dir
    args.inference = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_nl = load_ovr_model(model_nl_vs_others_path, device)
    model_n = load_ovr_model(model_n_vs_others_path, device)
    model_m = load_ovr_model(model_m_vs_others_path, device)


    # Train set
    train_dataset = FinetuneDataset(args.meta_file_path_train, args.bag_size, return_norm_factors=args.denorm)
    train_loader = DataLoader(
        train_dataset, batch_size=32,
        shuffle=False, num_workers=4, pin_memory=True, drop_last=False
    )

    train_embeds = extract_embeddings(model_nl, model_n, model_m, train_loader, device)
    train_embed_df = pd.DataFrame(train_embeds)
    print("train_embed_df.shape:", train_embed_df.shape)

    fpath = os.path.join(args.output_dir, "train_embeddings.parquet")
    train_embed_df.to_parquet(fpath, index=False)

    # Validation set
    valid_dataset = FinetuneDataset(args.meta_file_path_valid, args.bag_size, return_norm_factors=args.denorm)
    valid_loader = DataLoader(
        valid_dataset, batch_size=32,
        shuffle=False, num_workers=4, pin_memory=True, drop_last=False
    )

    valid_embeds = extract_embeddings(model_nl, model_n, model_m, valid_loader, device)
    valid_embed_df = pd.DataFrame(valid_embeds)
    print("valid_embed_df.shape:", valid_embed_df.shape)

    fpath = os.path.join(args.output_dir, "valid_embeddings.parquet")
    valid_embed_df.to_parquet(fpath, index=False)
