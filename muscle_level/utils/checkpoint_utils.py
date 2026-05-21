import torch
from pathlib import Path


def find_latest_checkpoint(output_dir):
    """체크포인트 폴더에서 가장 최근의 체크포인트 파일을 찾아서 반환"""
    ckpts_dir = Path(output_dir) / 'ckpts'
    if not ckpts_dir.exists():
        return None, None
    
    # backbone_{epoch:03d}.pth 패턴의 파일들을 찾기
    checkpoint_files = list(ckpts_dir.glob('backbone_*.pth'))
    if not checkpoint_files:
        return None, None
    
    # epoch 번호를 추출하여 정렬
    epochs_and_files = []
    for file in checkpoint_files:
        try:
            # 파일명에서 epoch 번호 추출 (backbone_000.pth -> 0)
            epoch_str = file.stem.split('_')[1]  # backbone_000 -> 000
            epoch = int(epoch_str)
            epochs_and_files.append((epoch, file))
        except (IndexError, ValueError):
            continue
    
    if not epochs_and_files:
        return None, None
    
    # epoch 번호로 정렬하여 가장 큰 epoch 반환
    epochs_and_files.sort(key=lambda x: x[0])
    latest_epoch, latest_file = epochs_and_files[-1]
    
    return latest_file, latest_epoch

def save_checkpoint(save_path, epoch, model, optimizer, loss_scaler=None, global_step=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    if loss_scaler is not None:
        checkpoint['scaler'] = loss_scaler.state_dict()
    
    if global_step is not None:
        checkpoint['global_step'] = global_step
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(weights_path, model=None, exclude_head=True, freeze=True, device='cpu'):
    """체크포인트를 로드합니다. model이 None이면 체크포인트 딕셔너리만 반환합니다."""
    checkpoint = torch.load(weights_path, weights_only=False, map_location=device)
    
    if model is None:
        return checkpoint
    
    # 기존 로직 (model이 제공된 경우)
    new_state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in new_state_dict.items()}

    matched_layers = 0
    unmatched_layers = []
    
    # MIL 관련 레이어들 (Backbone에만 있는 부분)
    mil_exclude_patterns = ['instance_attention']
    
    for name, param in model.state_dict().items():
        # MIL 관련 레이어 제외 (Backbone에만 있는 부분)
        if exclude_head and any(pattern in name for pattern in mil_exclude_patterns):
            unmatched_layers.append(name)
            continue
            
        if name in new_state_dict:
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
                # freeze=True면 인코더 고정, False면 학습 가능
                if param.dtype in [torch.float16, torch.float32, torch.float64]:
                    param.requires_grad = not freeze
            else:
                unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass # these are weights that weren't in the original model, such as a new head
            
    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")
    return model
