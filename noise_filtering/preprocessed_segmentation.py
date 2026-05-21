import click
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
import ray
import matplotlib.pyplot as plt

def save_muscle_data(output_file, muscle_data):  
    with h5py.File(output_file, 'w') as hf:
        patient_info = hf.create_group(f"patient_info")
        patient_info.attrs['pid'] = muscle_data[0]['pid']
        patient_info.attrs['visit_date'] = muscle_data[0]['visit_date']
        patient_info.attrs['sampling_rate'] = 9600
        patient_info.attrs['scale'] = "mv"
        patient_info.attrs['segment_num'] = len(muscle_data)

        emg_group = hf.create_group("emg")
        muscle_group = emg_group.create_group(f"muscle_{muscle_data[0]['muscle_index']}")
        
        muscle_info = muscle_group.create_group("muscle_info")
        muscle_info.attrs['lateral'] = muscle_data[0]['lateral']
        muscle_info.attrs['muscle'] = muscle_data[0]['muscle']

        for seg_data in muscle_data:
            seg_group = muscle_group.create_group(f"segment_{seg_data['segment_index']}")
            
            seg_info = seg_group.create_group("segment_info")
            seg_info.attrs['trace_index_list'] = seg_data['trace_index_list']
            
            seg_group.create_dataset('signal', data=seg_data['signal'], compression='gzip')



@ray.remote
def process_single_objectid(objectid, meta_data, save_dir):
    """단일 objectid를 처리하는 함수"""
    try:
        object_table = meta_data[meta_data['objectid']==objectid].copy()
        object_table.sort_values('segment_index', inplace=True)
        object_table.reset_index(drop=True, inplace=True)
        
        assert object_table['file_path'].nunique() == 1, f"objectid {objectid} has multiple file_paths"
        assert object_table['muscle_index'].nunique() == 1, f"objectid {objectid} has multiple muscle_indices"

        file_path = object_table.loc[0, "file_path"]
        muscle_index = object_table.loc[0, "muscle_index"]
        output_file = save_dir / f"{objectid}.h5"

        muscle_data = []
        
        # h5 파일을 한 번만 열어서 모든 segment 처리
        with h5py.File(file_path, 'r') as source_f:
            segment_group = object_table.groupby('segment_index')
            for segment_index, segment_table in segment_group:
                assert segment_table['segmentid'].nunique() == 1, f"segmentid has multiple segmentids"

                segment_table.sort_values('trace_index', inplace=True)
                segment_table.reset_index(drop=True, inplace=True)
                
                # segment signal 읽기
                signal_list = []
                for i, row in segment_table.iterrows():
                    signal = source_f[f"emg/muscle_{muscle_index}/segment_{row['trace_index']}/signal"][()]
                    signal_list.append(signal)
                segment_signal = np.concatenate(signal_list)
                
                assert segment_signal is not None and len(segment_signal) > 0, f"segmentid {objectid}_{segment_index} has no signal"

                segment_data = {
                    'segmentid': segment_table.loc[0, "segmentid"],
                    'pid' : segment_table.loc[0, "pid"],
                    'visit_date' : segment_table.loc[0, "visit_date"],
                    'muscle_index': muscle_index,
                    'lateral': segment_table.loc[0, "lateral"],
                    'muscle': segment_table.loc[0, "muscle"],
                    'segment_index': segment_index,
                    'trace_index_list' : segment_table['trace_index'].tolist(),
                    'signal': segment_signal,
                    'file_path': str(output_file)
                }
                muscle_data.append(segment_data)
        
        # 모든 segment 처리 완료 후 한 번에 저장
        save_muscle_data(str(output_file), muscle_data)

        filtered_muscle_data = [
            {k: v for k, v in seg.items() if k not in ['trace_index_list', 'signal']}
            for seg in muscle_data
        ]
        return filtered_muscle_data
    except Exception as e:
        print(f"Error processing objectid {objectid}: {str(e)}")
        return []


def plot_random_segments(save_dir, segment_meta_data, num_samples=500):
    print(f"Plotting {num_samples} random segments...")
    sampled = segment_meta_data.sample(n=min(num_samples, len(segment_meta_data)), random_state=42)
    
    for idx, row in tqdm(sampled.iterrows(), total=len(sampled), ncols=150):
        segmentid = row['segmentid']
        file_path = row['file_path']
        muscle_index = row['muscle_index']
        segment_index = row['segment_index']

        # h5 파일에서 signal 읽기
        try:
            with h5py.File(file_path, 'r') as f:
                signal = f[f"emg/muscle_{muscle_index}/segment_{segment_index}/signal"][()]
        except Exception as e:
            raise Exception(f"Error reading {file_path} segment {segment_index}: {e}")

        time = np.linspace(0, 400, len(signal))
        scale_list = [1, 5, 20]
        for scale in scale_list:
            plt.figure(figsize=(15, 5))
            plt.plot(time, signal)
            plt.title(f"{segmentid}")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (mV)")
            plt.grid()
            plt.ylim(-scale, scale)

            save_path = save_dir.parent / "sampled_segment_plots" / f"scale_{scale}"
            save_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path / f"{segmentid}.png", dpi=300, bbox_inches='tight')
            plt.close()

    print(f"{len(sampled)}개의 segment plot이 {save_dir}에 저장되었습니다.")



@click.command()
@click.option('--meta_data_path', type=str, default="/home/coder/workspace/data/BMI_nEMG/data/preprocessed/seg_400_meta_data.parquet", help='meta data path')
@click.option('--save-dir', type=str, default="/home/coder/workspace/data/BMI_nEMG/data/preprocessed/seg_400_h5", help='save directory')
def main(meta_data_path, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not Path(meta_data_path).exists():

        ray.init(
            num_cpus=70,
            _temp_dir="/home/coder/workspace/data/ray_temp"
        )
        
        meta_data = pd.read_parquet("/home/coder/workspace/data/BMI_nEMG/0_preprocessing/noise_filtering/outputs_v2/nosie_filtered_meta_data_400.parquet")
        meta_data['objectid'] = meta_data['pid'].astype(str) + '_' \
                                + meta_data['visit_date'].astype(str) + '_' \
                                + meta_data['muscle_index'].astype(str)
        objectid_list = meta_data['objectid'].unique().tolist()
        print(f"Processing {len(objectid_list)} objectids using Ray...")
        
        meta_data_ref = ray.put(meta_data)
        futures = []
        for objectid in objectid_list:
            future = process_single_objectid.remote(objectid, meta_data_ref, save_dir)
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
        segment_meta_data.to_parquet("/home/coder/workspace/data/BMI_nEMG/data/preprocessed/seg_400_meta_data.parquet", index=False)
        print(f"Total segments created: {len(segment_meta_data)}")
        print(segment_meta_data)

        plot_random_segments(save_dir, segment_meta_data, num_samples=500)
    else:
        segment_meta_data = pd.read_parquet(meta_data_path)
        plot_random_segments(save_dir, segment_meta_data, num_samples=500)


if __name__ == "__main__":
    main()
