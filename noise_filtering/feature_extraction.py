import os
import sys
from pathlib import Path
from ClusteringDataset import ClusteringDataset

import ray
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import linregress
import random
import pandas as pd
import click
import h5py
import joblib


@ray.remote
def extract_features_for_file(file_path, file_data, fs=9600, step=1):
    """
    Extract features for all segments in a given file.
    :param file_path: Path to the HDF5 file
    :param file_data: Segment data for this file (DataFrame)
    :param fs: Sampling frequency in Hz, default 9600
    :return: List of (segmentid, features) tuples
    """
    results = []
    
    with h5py.File(file_path, 'r') as f:
        for _, row in file_data.iterrows():
            muscle_index = row["muscle_index"]
            segment_index = row["segment_index"]
            segmentid = row["segmentid"]
            
            try:
                # Load signal
                signal = f[f"emg/muscle_{muscle_index}/segment_{segment_index}/signal"][()]
                signal = np.ascontiguousarray(signal).squeeze()

                # Extract features
                # features = extract_single_signal_features(signal, fs)
                features = extract_single_signal_features(signal, fs, step)
                results.append((segmentid, features))
                
            except Exception as e:
                print(f"Error processing segment {segmentid} from {file_path}: {e}")
                continue
    
    return results


def extract_features(table, step):
    print("Extract Features...")
    
    dataset = ClusteringDataset(table, inputs=[[{"signal":{"axis":["C","T"]}}]], targets=None)
    print(f"=> dataset size: {len(dataset)}")
    
    # Group by file_path
    file_groups = dataset.get_file_groups()
    print(f"=> Grouped into {len(file_groups)} files")

    # Create feature extraction tasks for all files
    file_refs = [extract_features_for_file.remote(file_path, file_data, step=step)
                for file_path, file_data in tqdm(file_groups, desc="Creating tasks", ncols=100)]

    # Collect results
    all_results = []
    with tqdm(total=len(file_refs), desc="Collecting results", ncols=100) as pbar:
        while file_refs:
            done_id, file_refs = ray.wait(file_refs, num_returns=min(10, len(file_refs)))
            batch_results = ray.get(done_id)
            all_results.extend(batch_results)
            pbar.update(len(done_id))

    # Process results - separate segmentid and features
    flattened_results = []
    for file_results in all_results:
        if isinstance(file_results, list):
            flattened_results.extend(file_results)
        else:
            flattened_results.append(file_results)

    segmentids = [result[0] for result in flattened_results]
    all_features = np.array([result[1] for result in flattened_results])

    print(f"=> Processed {len(segmentids)} segments in total")
    
    # Sort in order of segmentid
    sorted_indices = np.argsort(segmentids)
    sorted_features = all_features[sorted_indices]
    sorted_segmentids = np.array(segmentids)[sorted_indices]
    
    # Feature normalization
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(sorted_features)
    
    return scaled_features, sorted_segmentids, scaler


def extract_temporal_features(signal):
    # ----- Time domain features -----
    MAV = np.mean(np.abs(signal))
    STD = np.std(signal)
    WL = np.sum(np.abs(np.diff(signal)))
    ZC = np.sum(np.diff(np.sign(signal)) != 0)
    SSC = np.sum(np.diff(np.sign(np.diff(signal))) != 0)

    temporal_features = [MAV, STD, WL, ZC, SSC]
    return temporal_features

def extract_frequency_features(signal, fs=9600, step=1):
    # ----- Frequency domain features -----
    fft_vals = fft(signal)
    fft_freqs = fftfreq(len(signal), 1/fs)
    fft_power = np.abs(fft_vals)**2

    pos_mask = fft_freqs > 0
    freqs = fft_freqs[pos_mask]
    power = fft_power[pos_mask]

    MNF = np.sum(freqs * power) / np.sum(power) if np.sum(power) > 0 else 0

    cumulative_power = np.cumsum(power)
    total_power = cumulative_power[-1]
    MDF = freqs[np.where(cumulative_power >= total_power/2)][0] if total_power > 0 else 0

    PKF = freqs[np.argmax(power)] if len(power) > 0 else 0

    TP = np.sum(power)

    frequency_features = [MNF, MDF, PKF, TP]

    if step == 1:
        return frequency_features
    else:
        step2_features = extract_frequency_features_step2(freqs, power, TP, MNF)
        return [*frequency_features, *step2_features]

def extract_frequency_features_step2(freqs, power, TP, MNF):
    # 2. Compute per-band PSR (power spectral ratio relative to total power)
    bands = [
        (10, 40), (40, 60), (60, 80),
        (80, 100), (100, 150), (150, 200)
    ]
    psr_features = []
    for low, high in bands:
        band_mask = (freqs >= low) & (freqs < high)
        band_power = np.sum(power[band_mask])
        psr = band_power / TP if TP > 0 else 0
        psr_features.append(psr)
    
    # 3. Additional advanced features (optional)
    # Variance of center frequency (VCF)
    VCF = np.sum(power * (freqs - MNF)**2) / TP if TP > 0 else 0
    
    # Mean peak frequency
    peaks, _ = find_peaks(power)
    mean_peak_freq = np.mean(freqs[peaks]) if len(peaks) > 0 else 0
    
    # Spectral entropy
    power_normalized = power / TP if TP > 0 else np.zeros_like(power)
    spectral_entropy = -np.sum(power_normalized * np.log2(power_normalized + 1e-12))
    
    # Spectral slope
    valid_mask = (freqs > 0) & (power > 0)
    if np.any(valid_mask):
        log_freqs = np.log10(freqs[valid_mask])
        log_power = np.log10(power[valid_mask])
        slope, _, _, _, _ = linregress(log_freqs, log_power)
    else:
        slope = 0

    frequency_features = [*psr_features, VCF, mean_peak_freq, spectral_entropy, slope]
    return frequency_features

def extract_single_signal_features(signal, fs=9600, step=1):
    if step == 1:
        temporal_features = extract_temporal_features(signal)
        frequency_features = extract_frequency_features(signal, fs, step)
        return np.array([*temporal_features, *frequency_features])
    elif step == 2:
        frequency_features = extract_frequency_features(signal, fs, step)
        return np.array([*frequency_features])
    elif step == 3:
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        if signal_std > 0:
            signal = (signal - signal_mean) / signal_std
        temporal_features = extract_temporal_features(signal)
        frequency_features = extract_frequency_features(signal, fs, step)
        return np.array([*temporal_features, *frequency_features])
    else:
        raise ValueError(f"Invalid step: {step}")


@click.command()
@click.option('--cache-dir', type=str, default='./features')
@click.option('--sample-size', type=int, default=None, help='Data sampling size (None uses all data)')
@click.option('--step', type=int, default=1, help='Feature extraction step (1: time+freq, 2: freq only, 3: time+freq with normalization)')
@click.option('--existing-features', type=str, default=None, help='Path to an already-saved feature file; if provided, table is filtered by its segmentids') # "./features/all_step1_extracted_features.parquet", "./features/step2_extracted_features.parquet"
def main(cache_dir, sample_size, step, existing_features):
    random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    
    # Initialize Ray
    ray.init(
        num_cpus=50,
        _temp_dir="./data/ray_temp",
    )

    table = pd.read_parquet("./data/noise_filtered/clustering_meta_data.parquet")
    
    # If an existing feature file is present, filter table by its segmentids
    if existing_features is not None:
        existing_features_path = Path(existing_features)
        if existing_features_path.exists():
            print(f"Loading existing feature file and filtering table by segmentid: {existing_features}")
            existing_feature_df = pd.read_parquet(existing_features_path)
            existing_segmentids = existing_feature_df['segmentid'].values
            
            # Extract only samples matching the segmentids from the original table
            table = table[table['segmentid'].isin(existing_segmentids)].reset_index(drop=True)
            print(f"Filtered table size: {len(table)}")
        else:
            print(f"Warning: specified feature file does not exist: {existing_features}")
    
    if sample_size is not None: 
        table = table.sample(n=sample_size).reset_index(drop=True)

    scaled_features, segmentids, scaler = extract_features(table, step)
    cache_dir = Path(cache_dir) / f"step{step}_scaler.joblib"
    joblib.dump(scaler, str(cache_dir))

    print("Save Extracted Features...")
    feature_cols = [f'feature_{i}' for i in range(scaled_features.shape[1])]
    feature_df = pd.DataFrame(scaled_features, columns=feature_cols)
    feature_df['segmentid'] = segmentids
    feature_df = feature_df[['segmentid'] + feature_cols]

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(cache_dir / f"step{step}_extracted_features.parquet", index=False)
    print(feature_df.head())

    ray.shutdown()

if __name__ == '__main__':
    main()

    