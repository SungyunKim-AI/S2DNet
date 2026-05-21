import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

def _make_row_feature(group, i):
    """Build a single row feature vector identical to MILDataset (embed_nl, embed_n, anatomy)."""
    val_nl = group.iloc[i, 4]
    embed_nl = torch.as_tensor(np.array(val_nl, copy=True), dtype=torch.float32)
    embed_nl = embed_nl.unsqueeze(0)
    embed_nl = F.normalize(embed_nl, p=2, dim=1, eps=1e-10)
    val_n = group.iloc[i, 5]
    embed_n = torch.as_tensor(np.array(val_n, copy=True), dtype=torch.float32)
    embed_n = embed_n.unsqueeze(0)
    embed_n = F.normalize(embed_n, p=2, dim=1, eps=1e-10)
    val_anat = group.iloc[i, 6:].values.astype(np.float32)
    embed_anatomy = torch.as_tensor(val_anat).unsqueeze(0)
    return torch.cat([embed_nl, embed_n, embed_anatomy], dim=1).squeeze(0)


class MILDataset(Dataset):
    """Fixed bag size: zero-pad if fewer samples, randomly subsample if more. Returns padding_mask (True=real, False=padded)."""
    def __init__(self, df, bag_size=10, seed=42):
        self.bag_size = bag_size
        self.rng = np.random.default_rng(seed)
        self.label_map = {
            'normal': 0,
            'radiculopathy': 1,
            'focal neuropathy': 2,
            'polyneuropathy': 3,
            'systemic myopathy': 4
        }
        self.data_groups = []
        groups = df.groupby(['pid', 'visit_date'])
        for _, group in groups:
            label_str = group['subject_label'].iloc[0]
            if label_str not in self.label_map:
                continue
            self.data_groups.append((group, self.label_map[label_str]))

    def __len__(self):
        return len(self.data_groups)

    def __getitem__(self, idx):
        group, label = self.data_groups[idx]
        n = len(group)
        if n >= self.bag_size:
            indices = self.rng.choice(n, size=self.bag_size, replace=False)
            indices = np.sort(indices)
            features_list = [_make_row_feature(group, int(i)) for i in indices]
            padding_mask = torch.ones(self.bag_size, dtype=torch.bool)
        else:
            real = torch.stack([_make_row_feature(group, i) for i in range(n)], dim=0)
            pad = torch.zeros(self.bag_size - n, real.shape[-1], dtype=torch.float32)
            patient_features = torch.cat([real, pad], dim=0)
            padding_mask = torch.zeros(self.bag_size, dtype=torch.bool)
            padding_mask[:n] = True
            return patient_features, torch.tensor(label, dtype=torch.long), padding_mask
        patient_features = torch.stack(features_list, dim=0)
        padding_mask = torch.ones(self.bag_size, dtype=torch.bool)
        return patient_features, torch.tensor(label, dtype=torch.long), padding_mask

