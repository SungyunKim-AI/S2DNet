import click
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
import scipy.signal as scipy_signal
import ray

np.random.seed(42)
DATA_DIR = Path("/home/coder/workspace/data/BMI_nEMG/data/raw/subject_h5")
SAVE_DIR = Path("/home/coder/workspace/data/BMI_nEMG/data/noise_filtered/clustering_h5")


def read_total_signal(file_path, muscle_index, trace_table):
    signal_list = []
    for i, row in trace_table.iterrows():
        signal, sampling_rate, duration = read_trace_signal(file_path, muscle_index, row["trace_index"])
        signal = preprocess(signal, sampling_rate, duration)
        signal_list.append(signal)
    return np.concatenate(signal_list)


def read_trace_signal(file_path, muscle_index, trace_index):
    with h5py.File(file_path, 'r') as f:
        signal = f[f"emg/muscle_{muscle_index}/trace_{trace_index}/signal"][()]

        trace_info = f[f"emg/muscle_{muscle_index}/trace_{trace_index}/trace_info"]
        sampling_rate = trace_info.attrs.get('sampling_rate', None)
        duration = trace_info.attrs.get('duration', None)

    return signal, sampling_rate, duration


def preprocess(signal, sampling_rate, duration):
    if sampling_rate != 9600:
        num_samples = int(9600 * duration)
        signal = scipy_signal.resample(signal, num_samples)
    signal = signal / 1000
    return signal


def save_segment(output_file, muscle_data, segment_duration=0.1):  
    results = []
    with h5py.File(output_file, 'w') as hf:
        patient_info = hf.create_group(f"patient_info")
        patient_info.attrs['pid'] = muscle_data['pid']
        patient_info.attrs['visit_date'] = muscle_data['visit_date']
        patient_info.attrs['sampling_rate'] = 9600
        patient_info.attrs['scale'] = "mv"

        segment_length = int(9600 * segment_duration)
        segment_num = int(len(muscle_data['signal']) // segment_length)
        patient_info.attrs['segment_num'] = segment_num

        emg_group = hf.create_group("emg")
        muscle_group = emg_group.create_group(f"muscle_{muscle_data['muscle_index']}")
        
        muscle_info = muscle_group.create_group("muscle_info")
        muscle_info.attrs['lateral'] = muscle_data['lateral']
        muscle_info.attrs['muscle'] = muscle_data['muscle']

        for seg_idx in range(segment_num):
            seg_group = muscle_group.create_group(f"segment_{seg_idx}")
            seg_info = seg_group.create_group("segment_info")
            seg_start_index = seg_idx * segment_length
            seg_end_index = (seg_idx + 1) * segment_length
            seg_info.attrs['start_index'] = seg_start_index
            seg_info.attrs['end_index'] = seg_end_index
            seg_group.create_dataset('signal', data=muscle_data['signal'][seg_start_index:seg_end_index], compression='gzip')
            
            results.append({
                'pid':muscle_data['pid'], 'visit_date':muscle_data['visit_date'], 
                'objectid': muscle_data['objectid'], 'segmentid': f"{muscle_data['objectid']}_{seg_idx}",
                'muscle_index': muscle_data['muscle_index'], 'lateral': muscle_data['lateral'], 'muscle':muscle_data['muscle'],
                'segment_index':seg_idx, 'start_index':seg_start_index, 'end_index':seg_end_index, 'file_path':output_file
            })
        
    return results


@ray.remote
def process_single_objectid(objectid, meta_data, save_dir):
    """Process a single objectid"""
    try:
        trace_table = meta_data[meta_data['objectid']==objectid]
        trace_table = trace_table.sort_values('trace_index')
        trace_table.reset_index(drop=True, inplace=True)

        file_path = trace_table.loc[0, "file"]
        muscle_index = trace_table.loc[0, "muscle_index"]

        muscle_signal = read_total_signal(file_path, muscle_index, trace_table)
        if muscle_signal is None or len(muscle_signal) == 0:
            print(f"Warning: signal is none for objectid {objectid}")
            return []

        muscle_data = {
            'pid' : trace_table.loc[0, "pid"],
            'visit_date' : trace_table.loc[0, "visit_date"],
            'objectid': objectid,
            'muscle_index': muscle_index,
            'lateral': trace_table.loc[0, "lateral"],
            'muscle': trace_table.loc[0, "muscle"],
            'signal': muscle_signal
        }
        output_file = save_dir / f"{objectid}.h5"
        segment_results = save_segment(str(output_file), muscle_data)
        return segment_results
    except Exception as e:
        print(f"Error processing objectid {objectid}: {str(e)}")
        return []

@click.command()
@click.option('--sample-size', '-s', default=None, help='Number of random samples to select (default: 10000)')
def main(sample_size):
    ray.init(
        num_cpus=50,
        _temp_dir="/home/coder/workspace/data/ray_temp"
    )
    
    meta_data = pd.read_parquet("/home/coder/workspace/data/BMI_nEMG/data/raw/subject_h5/filtered_meta_data.parquet")
    meta_data = meta_data[meta_data['sampling_rate'] == 9600]
    meta_data = meta_data[meta_data['duration'] == 0.1]
    meta_data = meta_data[meta_data['notch_filter'] == 60]
    meta_data = meta_data[meta_data['low_filter'] <= 4]
    meta_data = meta_data[(meta_data['high_filter'] == -1) | (meta_data['high_filter'] == 5000)]

    meta_data['objectid'] = meta_data['pid'].astype(str) + '_' \
                            + meta_data['visit_date'].astype(str) + '_' \
                            + meta_data['muscle_index'].astype(str)
    meta_data['file'] = meta_data['file'].apply(lambda x : str(DATA_DIR/x))

    objectid_list = meta_data['objectid'].unique().tolist()
    print("objectid num:", len(objectid_list))

    if sample_size is not None:
        # If sample_size exceeds total number of objectids, use all
        print(f"Selecting {sample_size} random samples...")
        actual_sample_size = min(sample_size, len(objectid_list))
        oids = np.random.choice(objectid_list, size=actual_sample_size, replace=False)
    else:
        oids = objectid_list.copy()
    print(f"Processing {len(oids)} objectids using Ray...")
    
    meta_data_ref = ray.put(meta_data)
    futures = []
    for objectid in oids:
        future = process_single_objectid.remote(objectid, meta_data_ref, SAVE_DIR)
        futures.append(future)
    
    total_results = []
    remaining_futures = futures.copy()
    with tqdm(total=len(futures), ncols=150, desc="Processing get results") as pbar:
        while remaining_futures:
            ready_futures, remaining_futures = ray.wait(remaining_futures, num_returns=1)
            
            for future in ready_futures:
                result = ray.get(future)
                total_results.extend(result)
                pbar.update(1)


    ray.shutdown()

    segment_meta_data = pd.DataFrame(total_results)
    segment_meta_data = segment_meta_data.sort_values(['pid', 'visit_date', 'muscle_index', 'segment_index'])
    segment_meta_data.reset_index(drop=True, inplace=True)
    segment_meta_data.to_parquet("/home/coder/workspace/data/BMI_nEMG/data/noise_filtered/clustering_meta_data.parquet", index=False)
    print(f"Total segments created: {len(segment_meta_data)}")
    print(segment_meta_data)


if __name__ == "__main__":
    main()