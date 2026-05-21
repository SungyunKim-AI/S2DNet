import h5py
import numpy as np
import pandas as pd
import torch
import torch.fft as fft
from torch.utils.data import Dataset, DataLoader, DistributedSampler


def normalize_and_get_stats(tensor):
    mean = tensor.mean(axis=-1, keepdims=True)
    std = tensor.std(axis=-1, keepdims=True)
    normalized_tensor = (tensor - mean) / (std + 1e-8)
    return normalized_tensor, mean, std


class PretrainDataset(Dataset):

    def __init__(self, meta_file):
        super(PretrainDataset, self).__init__()
        if isinstance(meta_file, pd.DataFrame):
            self.meta_data = meta_file
        else:
            self.meta_data = pd.read_parquet(meta_file)
        self.segment_list = self.meta_data['segmentid'].tolist()


    def _load_signal(self, file_path, muscle_index, segment_index):
        try:
            with h5py.File(file_path, 'r') as f:
                signal = f[f"emg/muscle_{muscle_index}/segment_{segment_index}/signal"][()]
        except Exception as e:
            raise Exception(f"Error reading {file_path} segment {segment_index}: {e}")
        signal, mean, std = self._preprocessing(signal)
        signal = signal.reshape(1,-1)
        return signal, mean, std

    def _preprocessing(self, signal):
        signal = np.clip(signal, -20, 20)   # clip to -20 / 20
        mean = np.mean(signal)
        std = np.std(signal)
        signal = (signal - mean) / (std + 1e-8)
        return signal, mean, std

    def _get_data_dict(self, data_time):
        """공통 데이터 처리 로직을 별도 메서드로 분리"""
        data_freq = fft.fft(data_time).abs()
        data_freq, _, _ = normalize_and_get_stats(data_freq)
        
        len_f = int(0.5 * data_freq.shape[1])
        data_dict = {'time': data_time, 'freq': data_freq[:, :len_f]}
        return data_dict

    def __getitem__(self, index):
        data = self.meta_data.iloc[index]
        data_time, _, _ = self._load_signal(data['file_path'], data['muscle_index'], data['segment_index'])
        data_time = torch.tensor(data_time.copy(), dtype=torch.float)

        data_dict = self._get_data_dict(data_time)
        return data_dict

    def __len__(self):
        return len(self.segment_list)


def get_pretraining_dataloaders(train_meta, val_meta, world_size, rank, batch_size):
    train_dataset = PretrainDataset(train_meta)
    val_dataset = PretrainDataset(val_meta)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=4, pin_memory=True, drop_last=False
    )

    return train_loader, val_loader


class FinetuneDataset(PretrainDataset):
    def __init__(self, meta_file, bag_size=50, return_norm_factors=False):
        super(FinetuneDataset, self).__init__(meta_file)
        self.label_map = {'nl': 0, 'n': 1, 'm': 2}
        self.bag_size = bag_size
        self.return_norm_factors = return_norm_factors
        
        # pid, visit_date, muscle_index로 그룹화
        self.grouped = self.meta_data.groupby(['pid', 'visit_date', 'muscle_index'])
        self.group_keys = list(self.grouped.groups.keys())
    
    def __len__(self):
        return len(self.group_keys)
    
    def __getitem__(self, index):
        group_key = self.group_keys[index]
        data = self.grouped.get_group(group_key).reset_index(drop=True)
        
        actual_size = len(data)
        collect_size = min(actual_size, self.bag_size)
        
        data_time_bag, data_freq_bag = [], []
        mean_bag, std_bag = [], []
        for i in range(collect_size):
            row = data.iloc[i]
            data_time, mean, std = self._load_signal(row['file_path'], row['muscle_index'], row['segment_index'])
            data_time = torch.tensor(data_time.copy(), dtype=torch.float)
            data_time_bag.append(data_time)

            data_freq = self._get_data_dict(data_time)['freq']
            data_freq_bag.append(data_freq)

            # numpy 값을 torch tensor로 변환
            mean_bag.append(torch.tensor(mean, dtype=torch.float))
            std_bag.append(torch.tensor(std, dtype=torch.float))
        
        # 패딩 (50개 미만인 경우)
        if actual_size < self.bag_size:
            pad_size = self.bag_size - actual_size
            zero_time = torch.zeros_like(data_time_bag[0])
            zero_freq = torch.zeros_like(data_freq_bag[0])
            
            for _ in range(pad_size):
                data_time_bag.append(zero_time.clone())
                data_freq_bag.append(zero_freq.clone())
                mean_bag.append(torch.tensor(0.0, dtype=torch.float))
                std_bag.append(torch.tensor(1.0, dtype=torch.float))
        
        data_dict = {
            'time': torch.stack(data_time_bag, dim=0),  # (bag_size, 1, seq_len)
            'freq': torch.stack(data_freq_bag, dim=0)   # (bag_size, 1, seq_len//2)
        }

        if self.return_norm_factors:
            norm_dict = {
                'mean': torch.stack(mean_bag, dim=0).unsqueeze(-1),   # (bag_size, 1)
                'std': torch.stack(std_bag, dim=0).unsqueeze(-1)      # (bag_size, 1)
            }
        else:
            norm_dict = {
                'mean': torch.zeros(self.bag_size, 1),
                'std': torch.ones(self.bag_size, 1)
            }

        label = data.iloc[0]['label']
        data_label = torch.tensor(self.label_map[label], dtype=torch.long)

        return data_dict, data_label, norm_dict, group_key


class FinetuneDataset_ovr(FinetuneDataset):
    """앙상블용 이진 분류를 위한 FinetuneDataset (one-vs-others 방식)"""
    
    def __init__(self, meta_file, bag_size=50, return_norm_factors=False, binary_type='0_vs_others'):
        """
        binary_type: '0_vs_others' (nl vs n+m), '1_vs_others' (n vs nl+m), '2_vs_others' (m vs nl+n)
        """
        super(FinetuneDataset_ovr, self).__init__(meta_file, bag_size, return_norm_factors)
        self.binary_type = binary_type
        
        # binary_type에 따라 라벨 매핑: 대상 클래스=1(positive), 나머지=0
        self.valid_labels = ['nl', 'n', 'm']
        if binary_type == 'nl_vs_others':
            self.label_map = {'nl': 1, 'n': 0, 'm': 0}
        elif binary_type == 'n_vs_others':
            self.label_map = {'n': 1, 'nl': 0, 'm': 0}
        elif binary_type == 'm_vs_others':
            self.label_map = {'m': 1, 'nl': 0, 'n': 0}
        self._filter_data()
    
    def _filter_data(self):
        """binary_type에 따라 데이터 필터링"""
        self.meta_data = self.meta_data[self.meta_data['label'].isin(self.valid_labels)].reset_index(drop=True)
        self.grouped = self.meta_data.groupby(['pid', 'visit_date', 'muscle_index'])
        self.group_keys = list(self.grouped.groups.keys())
    
    def __getitem__(self, index):
        group_key = self.group_keys[index]
        data = self.grouped.get_group(group_key).reset_index(drop=True)
        
        actual_size = len(data)
        collect_size = min(actual_size, self.bag_size)
        
        data_time_bag, data_freq_bag = [], []
        mean_bag, std_bag = [], []
        for i in range(collect_size):
            row = data.iloc[i]
            data_time, mean, std = self._load_signal(row['file_path'], row['muscle_index'], row['segment_index'])
            data_time = torch.tensor(data_time.copy(), dtype=torch.float)
            data_time_bag.append(data_time)

            data_freq = self._get_data_dict(data_time)['freq']
            data_freq_bag.append(data_freq)

            mean_bag.append(torch.tensor(mean, dtype=torch.float))
            std_bag.append(torch.tensor(std, dtype=torch.float))
        
        # 패딩 (50개 미만인 경우)
        if actual_size < self.bag_size:
            pad_size = self.bag_size - actual_size
            zero_time = torch.zeros_like(data_time_bag[0])
            zero_freq = torch.zeros_like(data_freq_bag[0])
            
            for _ in range(pad_size):
                data_time_bag.append(zero_time.clone())
                data_freq_bag.append(zero_freq.clone())
                mean_bag.append(torch.tensor(0.0, dtype=torch.float))
                std_bag.append(torch.tensor(1.0, dtype=torch.float))
        
        data_dict = {
            'time': torch.stack(data_time_bag, dim=0),  # (bag_size, 1, seq_len)
            'freq': torch.stack(data_freq_bag, dim=0)   # (bag_size, 1, seq_len//2)
        }

        if self.return_norm_factors:
            norm_dict = {
                'mean': torch.stack(mean_bag, dim=0).unsqueeze(-1),   # (bag_size, 1)
                'std': torch.stack(std_bag, dim=0).unsqueeze(-1)      # (bag_size, 1)
            }
        else:
            norm_dict = {
                'mean': torch.zeros(self.bag_size, 1),
                'std': torch.ones(self.bag_size, 1)
            }

        label = data.iloc[0]['label']
        data_label = torch.tensor(self.label_map[label], dtype=torch.long)

        return data_dict, data_label, norm_dict, group_key

def calculate_pos_weights(dataset):
    """
    Calculate positive weights for BCEWithLogitsLoss based on class imbalance.
    pos_weight = negative_samples / positive_samples
    This is calculated per class (OvR context).
    """
    print("Calculating positive weights for OvR loss...")
    if hasattr(dataset, 'meta_data'):
        # Group by PID/Date/Muscle to count actual samples (bags)
        # Assuming each group has consistent label
        grouped_labels = dataset.meta_data.groupby(['pid', 'visit_date', 'muscle_index'])['label'].first()
        label_counts = grouped_labels.value_counts().to_dict()
        
        label_map = dataset.label_map # e.g., {'nl': 0, 'n': 1, 'm': 2}
        num_classes = len(label_map)
        
        pos_weights = torch.ones(num_classes)
        total_samples = len(grouped_labels)
        
        print(f"Total samples: {total_samples}")
        print("Class distribution & Weights:")
        for label_str, class_idx in label_map.items():
            n_pos = label_counts.get(label_str, 0)
            n_neg = total_samples - n_pos
            
            if n_pos > 0:
                weight = n_neg / n_pos
            else:
                weight = 1.0 # Fallback
                
            pos_weights[class_idx] = weight
            print(f"  Class '{label_str}' ({class_idx}): {n_pos} positive, {n_neg} negative -> pos_weight={weight:.4f}")
            
        return pos_weights
    else:
        print("Warning: Dataset does not have meta_data attribute or logic is not applicable.")
        return None
