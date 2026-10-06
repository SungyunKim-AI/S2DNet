import numpy as np
import pandas as pd
import torch
import h5py
import logging
import traceback as tb
from torch.utils.data import Dataset


class ClusteringDataset(Dataset):
    def __init__(
        self, table, inputs=["signal"], targets=[{"label":None}], stage="test", dataset_params=None
    ):
        self.table = table
        self.len = len(table)
        self.stage = stage
        self.inputs = inputs
        self.targets = targets
        self.success_idx = 0

        self.dataset_params = dataset_params

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        try:
            objectid, signal, label = self._read_signal(index)
            self.success_idx = index
        except Exception:
            logging.getLogger().error(
                f"Failed to _read_signal in Dataset: {tb.format_exc()}"
            )
            objectid, signal, label = self._read_signal(self.success_idx)

        if label is None:
            return objectid, signal
        else:
            return objectid, signal, label

    def _read_signal(self, index):
        row = self.table.iloc[index]
        file_path = row["file_path"]
        muscle_index, segment_index = row["muscle_index"], row["segment_index"]
        
        # Inputs 
        for input_group in self.inputs:
            for input_dict in input_group:
                for input_key in input_dict.keys():
                    if input_key == "signal":
                        signal_tensor = self._load_signal(file_path, muscle_index, segment_index)  # (Channel, Length)

        # Targets 
        if self.targets is not None:
            for target_dict in self.targets:
                for target_key, target_meta in target_dict.items():
                    if target_key in ["*"]:
                        target_tensor = torch.tensor([[]], dtype=torch.long)
                    else:
                        value = self._get_value(row, target_key)
                        target_tensor = torch.tensor([[value]], dtype=torch.long)
            return row["segmentid"], signal_tensor, target_tensor
        else:
            return row["segmentid"], signal_tensor, None

    def _get_value(self, row, key, default=np.nan):
        if key in row and pd.notna(row[key]):
            return row[key]

    def _load_signal(self, file_path, muscle_index, segment_index):
        with h5py.File(file_path, 'r') as f:
            signal = f[f"emg/muscle_{muscle_index}/segment_{segment_index}/signal"][()]
        signal = np.ascontiguousarray(signal)
        signal = self._preprocessing(signal)
        signal = signal.reshape(1, -1)
        return torch.tensor(signal, dtype=torch.float32)
    
    def _preprocessing(self, signal):
        signal = np.clip(signal, -20, 20)   # clip to -20 / 20
        signal = signal - np.mean(signal)   # DC offset
        return signal

    def get_file_groups(self):
        """Return data grouped by file_path."""
        return self.table.groupby("file_path")

    def get_segments_for_file(self, file_path, file_data):
        """Load all segment data for a specific file at once."""
        segments_data = []
        
        with h5py.File(file_path, 'r') as f:
            for _, row in file_data.iterrows():
                muscle_index = row["muscle_index"]
                segment_index = row["segment_index"]
                segmentid = row["segmentid"]
                
                try:
                    signal = f[f"emg/muscle_{muscle_index}/segment_{segment_index}/signal"][()]
                    signal = np.ascontiguousarray(signal)
                    signal = self._preprocessing(signal)
                    signal = signal.reshape(1, -1)
                    signal_tensor = torch.tensor(signal, dtype=torch.float32)
                    
                    segments_data.append({
                        'segmentid': segmentid,
                        'signal': signal_tensor,
                        'row_data': row
                    })
                except Exception as e:
                    print(f"Error loading segment {segmentid} from {file_path}: {e}")
                    continue
        
        return segments_data


if __name__ == "__main__":
    table = pd.read_parquet("./data/noise_filtered/clustering_meta_data.parquet")
    dataset = ClusteringDataset(table, inputs=[[{"signal":{"axis":["C","T"]}}]], targets=None)
    
    segmentid, signals = dataset[0]
    signal = signals[0].numpy().squeeze()
    print(segmentid)
    print(signals.shape)
    print(len(signal))
