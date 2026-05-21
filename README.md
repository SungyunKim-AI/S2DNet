# S2DNet

**S2D-Net (Signal to Diagnosis Network)** — an end-to-end pipeline for automated neuromuscular disease diagnosis from needle EMG (nEMG) signals.

The pipeline consists of three sequential stages:

```
Raw EMG (HDF5)
    │
    ▼
[1. noise_filtering]   Segment raw traces → extract features → cluster noise
    │
    ▼
[2. muscle_level]      Pretrain BimodalMAE → fine-tune One-vs-Rest classifiers
    │
    ▼
[3. patient_level]     Extract muscle embeddings → train patient-level MIL classifier
```

---

## 1. noise_filtering

Processes raw HDF5 EMG recordings into clean, labeled segments ready for model training. The stage runs a two-step clustering pipeline (coarse → fine) using Apache Spark and Ray.

### Scripts

#### `signal_segmentation_100ms.py`
Reads raw EMG traces, resamples to 9600 Hz, and segments into 100 ms windows. Saves each muscle object as an HDF5 file and produces a metadata parquet.

```bash
python signal_segmentation_100ms.py [--sample-size N]
```

| Option | Default | Description |
|---|---|---|
| `--sample-size` | None (all) | Number of random objectids to process |

**Output:** `clustering_h5/*.h5`, `clustering_meta_data.parquet`

---

#### `feature_extraction.py`
Extracts time-domain and frequency-domain features from each segment in parallel via Ray. Three extraction modes are supported.

```bash
python feature_extraction.py \
  --cache-dir ./features \
  --step 1 \
  [--sample-size N] \
  [--existing-features PATH]
```

| Option | Default | Description |
|---|---|---|
| `--cache-dir` | `./features` | Directory for output feature files |
| `--step` | `1` | `1`: time+freq, `2`: freq only, `3`: time+freq (normalized) |
| `--sample-size` | None | Limit number of segments |
| `--existing-features` | None | Filter table to segmentids found in this parquet |

**Output:** `features/step{N}_extracted_features.parquet`, `step{N}_scaler.joblib`

---

#### `find_n_cluster.py`
Scans a range of k values using Spark KMeans, computes inertia and silhouette scores, and plots an elbow curve to guide cluster count selection.

```bash
python find_n_cluster.py \
  --cache-dir ./features \
  --k-min 2 --k-max 10 \
  [--auto-k] \
  [--clustering-result-path PATH --target-clusters 0,3]
```

| Option | Default | Description |
|---|---|---|
| `--k-min` / `--k-max` | 2 / 10 | Range of cluster counts to test |
| `--auto-k` | False | Auto-select optimal k via elbow method |
| `--sample-size` | None | Downsample before fitting |
| `--clustering-result-path` | — | Filter features by a prior clustering result |
| `--target-clusters` | — | Comma-separated cluster IDs to keep |

**Output:** `elbow_curve.png`, `kmeans_clustering_analysis.csv`

---

#### `clustering_spark.py`
Runs large-scale KMeans or GMM clustering using Spark. Supports chaining from a previous clustering step to refine noisy clusters.

```bash
python clustering_spark.py \
  --cache-dir ./features \
  --method kmeans \
  --n-clusters 5 \
  --step 2 \
  [--clustering-result-path PATH --target-clusters 0,4,6]
```

| Option | Default | Description |
|---|---|---|
| `--method` | `kmeans` | `kmeans`, `gmm`, or `all` |
| `--n-clusters` | `5` | Number of clusters |
| `--step` | `2` | Feature step to load |
| `--clustering-result-path` | — | Filter by prior clustering result |
| `--target-clusters` | — | Cluster IDs to use from prior result |

**Output:** `outputs_v2/step{N}_kmeans/clustering_result.parquet`, saved model, cluster centers

---

#### `inference_spark.py`
Applies a previously saved Spark KMeans model to new features. Useful for assigning cluster labels to held-out or production data.

```bash
python inference_spark.py \
  --model-path ./outputs/step1_kmeans/kmeans_model \
  --features-path ./features/step2_extracted_features.parquet \
  --output-path ./outputs/inference/step2_kmeans \
  [--clustering-result-path PATH --target-clusters 0,5]
```

**Output:** `clustering_result.parquet` with `segmentid` and `prediction` columns

---

#### `preprocessed_segmentation.py`
Reorganizes noise-filtered segments into structured HDF5 files grouped by patient/visit/muscle. Optionally plots random segment waveforms for quality inspection.

```bash
python preprocessed_segmentation.py \
  --meta_data_path PATH/seg_400_meta_data.parquet \
  --save-dir PATH/seg_400_h5
```

**Output:** `seg_400_h5/*.h5`, `seg_400_meta_data.parquet`

---

#### `ClusteringDataset.py`
PyTorch `Dataset` that reads HDF5 segment signals grouped by file path. Used internally by `feature_extraction.py`.

---

## 2. muscle_level

Trains a **BimodalMAE** (Bimodal Masked Autoencoder) encoder on unlabeled segments, then fine-tunes three binary One-vs-Rest (OvR) classifiers for Normal / Neuropathy / Myopathy detection using Multi-Instance Learning (MIL).

### Model architecture

- **Input modalities:** time-domain signal + frequency-domain (FFT magnitude)
- **Encoder:** Vision Transformer with Adaptive Spectral Block, multi-scale multi-channel processing
- **Decoder:** Cross-attention reconstruction adapter with gated inter-layer skip connections
- **Classification head:** `MILOutputAdapter` — learnable bag query + multi-head attention aggregation over bag instances

### Scripts

#### `pretrain.py`
Self-supervised pretraining via masked reconstruction. Both time and frequency branches are masked independently, then reconstructed. Supports multi-GPU DDP.

```bash
python pretrain.py \
  -c configs/pretrain.yaml \
  --exp_name BimodalMAE_backbone \
  --output_dir outputs/pretrain \
  --meta_file_path_train TRAIN.parquet \
  --meta_file_path_valid VALID.parquet
```

Key hyperparameters (overridable via CLI):

| Option | Default | Description |
|---|---|---|
| `--seq_len` | `3840` | Signal length (samples) |
| `--patch_size` | `16` | Patch size for tokenization |
| `--mask_ratio_t` / `--mask_ratio_f` | `0.5` | Masking ratio for time / frequency |
| `--epochs` | `200` | Training epochs |
| `--batch_size` | `1024` | Per-GPU batch size |
| `--lr` | `1e-4` | Base learning rate |
| `--task_balancer` | `none` | Loss weighting: `none` or `uncertainty` |

**Config:** `configs/pretrain.yaml`  
**Output:** `outputs/pretrain/ckpts/backbone_*.pth`, `config.yaml`

---

#### `finetuning.py`
Fine-tunes the pretrained encoder for one binary OvR task. Run separately for each of the three binary types.

```bash
# Normal vs Others
python finetuning.py -c configs/finetune_ovr.yaml \
  --binary_type nl_vs_others \
  --pretrained_config_path outputs/pretrain/config.yaml \
  --pretrained_weights_path outputs/pretrain/ckpts/backbone_best.pth \
  --meta_file_path_train TRAIN.parquet \
  --meta_file_path_valid VALID.parquet

# Neuropathy vs Others
python finetuning.py -c configs/finetune_ovr.yaml --binary_type n_vs_others ...

# Myopathy vs Others
python finetuning.py -c configs/finetune_ovr.yaml --binary_type m_vs_others ...
```

Key options:

| Option | Default | Description |
|---|---|---|
| `--binary_type` | `nl_vs_others` | `nl_vs_others`, `n_vs_others`, `m_vs_others` |
| `--bag_size` | `30` | Instances per MIL bag |
| `--denorm` | `True` | Apply denormalization in forward pass |
| `--epochs` | `50` | Fine-tuning epochs |
| `--lr` | `0.001` | Learning rate |
| `--criterion` | `CrossEntropyLoss` | Loss function (`CrossEntropyLoss` or `FocalLoss`) |

**Config:** `configs/finetune_ovr.yaml`  
**Output:** One checkpoint per binary type in `--output_dir`

---

#### `inference.py`
Loads all three trained OvR checkpoints and runs ensemble inference on a validation set.

```bash
python inference.py \
  -c configs/finetune_ensemble.yaml \
  --meta_file_path_valid VALID.parquet \
  --model_nl_vs_others_path PATH/nl_vs_others/best.pth \
  --model_n_vs_others_path  PATH/n_vs_others/best.pth \
  --model_m_vs_others_path  PATH/m_vs_others/best.pth \
  --output_dir outputs/muscle_level/ensemble
```

**Config:** `configs/finetune_ensemble.yaml`  
**Output:** Per-muscle predictions, AUROC/AUPRC metrics, summary report

---

### Directory layout

```
muscle_level/
├── pretrain.py
├── finetuning.py
├── inference.py
├── configs/
│   ├── pretrain.yaml
│   ├── finetune_ovr.yaml
│   └── finetune_ensemble.yaml
├── models/
│   ├── networks.py          # BimodalMAE, BimodalMAE_Classifier
│   ├── input_adapters.py    # Time/frequency patch embedding
│   ├── output_adapters.py   # Reconstruction & MIL output heads
│   └── models_utils.py      # Transformer blocks, utilities
└── utils/
    ├── datasets.py          # PretrainDataset, FinetuneDataset, FinetuneDataset_ovr
    ├── checkpoint_utils.py  # Save/load/find checkpoints
    ├── get_loss.py          # MaskedMSELoss, FocalLoss
    ├── logger.py            # TensorBoard logger, MetricLogger
    ├── native_scaler.py     # AMP scaler, cosine scheduler
    ├── optim_factory.py     # AdamW with layer-wise LR decay
    └── task_balancing.py    # NoWeighting, UncertaintyWeighting
```

---

## 3. patient_level

Aggregates per-muscle OvR embeddings across a patient visit into a fixed bag and trains a **Multi-Expert Gated MIL** classifier to predict the patient-level diagnosis (5 classes: normal, radiculopathy, focal neuropathy, polyneuropathy, systemic myopathy).

### Scripts

#### `extract_muscle_embedding.py`
Loads the three trained OvR models and extracts the bag-level CLS token embedding for every muscle bag in the dataset. Embeddings from the three models are L2-normalized and concatenated: `[embed_nl ‖ embed_n ‖ embed_m]`.

```bash
python extract_muscle_embedding.py
```

Edit the script's path constants to point to the OvR checkpoint folders and parquet files.

**Output:** `BimodalMAE_train_embed.parquet`, `BimodalMAE_valid_embed.parquet`  
Columns: `pid`, `visit_date`, `muscle_index`, `embed_nl`, `embed_n`, anatomy features

---

#### `train_mil.py`
Trains the `MultiExpertGatedMIL` model on the extracted embeddings.

```bash
python train_mil.py \
  --train_data PATH/train_embed.parquet \
  --valid_data PATH/valid_embed.parquet \
  --output_dir outputs/gated_attention_mil_v1 \
  --gpu 0
```

Key options:

| Option | Default | Description |
|---|---|---|
| `--muscle_dim` | `192` | Embedding dimension per OvR model |
| `--hidden_dim` | `128` | Hidden dimension inside experts |
| `--num_classes` | `4` | Number of diagnosis classes |
| `--num_heads` | `4` | Attention heads |
| `--num_layers` | `2` | Transformer layers |
| `--ablation_mode` | `full` | `full`, `no_muscle`, `no_anatomy` |
| `--bag_size` | `10` | Fixed bag size (0 = variable) |
| `--batch_size` | `256` | Training batch size |
| `--num_epochs` | `50` | Max epochs |
| `--learning_rate` | `1e-4` | Initial learning rate |
| `--focal_gamma` | `2.0` | Focal loss gamma |
| `--patience` | `25` | Early stopping patience |

**Output:** Model checkpoint, per-class AUROC/AUPRC metrics, training report

---

### Directory layout

```
patient_level/
├── extract_muscle_embedding.py
├── train_mil.py
├── dataset.py               # MILDataset — fixed bag with zero-padding
├── models/
│   └── multi_expert_gated_mil.py   # MultiExpertGatedMIL
└── utils/
    ├── get_loss.py          # FocalLoss
    └── get_metric.py        # AUROC, AUPRC, metrics saving
```

---

## Requirements

- Python 3.10
- PyTorch 2.9
- Apache Spark (PySpark) — for noise_filtering clustering steps
- Ray — for parallel feature extraction and segmentation
- `h5py`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `joblib`
- `einops`, `timm`, `click`, `tqdm`, `matplotlib`
