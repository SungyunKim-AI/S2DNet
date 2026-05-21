import torch
from pathlib import Path


def find_latest_checkpoint(output_dir):
    """Find and return the most recent checkpoint file in the checkpoint folder"""
    ckpts_dir = Path(output_dir) / 'ckpts'
    if not ckpts_dir.exists():
        return None, None
    
    # Find files matching the pattern backbone_{epoch:03d}.pth
    checkpoint_files = list(ckpts_dir.glob('backbone_*.pth'))
    if not checkpoint_files:
        return None, None
    
    # Extract epoch numbers and sort
    epochs_and_files = []
    for file in checkpoint_files:
        try:
            # Extract epoch number from filename (backbone_000.pth -> 0)
            epoch_str = file.stem.split('_')[1]  # backbone_000 -> 000
            epoch = int(epoch_str)
            epochs_and_files.append((epoch, file))
        except (IndexError, ValueError):
            continue
    
    if not epochs_and_files:
        return None, None
    
    # Sort by epoch number and return the largest epoch
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
    """Load a checkpoint. If model is None, returns only the checkpoint dictionary."""
    checkpoint = torch.load(weights_path, weights_only=False, map_location=device)
    
    if model is None:
        return checkpoint
    
    # Existing logic (when model is provided)
    new_state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in new_state_dict.items()}

    matched_layers = 0
    unmatched_layers = []
    
    # MIL-related layers (only present in Backbone)
    mil_exclude_patterns = ['instance_attention']
    
    for name, param in model.state_dict().items():
        # Exclude MIL-related layers (only present in Backbone)
        if exclude_head and any(pattern in name for pattern in mil_exclude_patterns):
            unmatched_layers.append(name)
            continue
            
        if name in new_state_dict:
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
                # If freeze=True, encoder is frozen; if False, encoder is trainable
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
