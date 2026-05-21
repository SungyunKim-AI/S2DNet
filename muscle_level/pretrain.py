import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import yaml
import argparse
from pathlib import Path
from typing import Dict, Iterable
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Import from utils
from utils.datasets import PretrainDataset
from utils.checkpoint_utils import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.get_loss import MaskedMSELoss
from utils.optim_factory import create_optimizer
from utils.native_scaler import NativeScalerWithGradNormCount as NativeScaler
from utils.task_balancing import NoWeightingStrategy, UncertaintyWeightingStrategy
from utils.logger import TensorBoardLogger, MetricLogger, SmoothedValue
from utils.native_scaler import cosine_scheduler

# Import from models
from models.networks import BimodalMAE
from models.input_adapters import PatchedInputAdapter
from models.output_adapters import ReConstructOutputAdapter


def setup_ddp(rank, world_size, backend='nccl'):
    """DDP 초기화"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 프로세스 그룹 초기화
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # GPU 설정
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """DDP 정리"""
    dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser('BimodalMAE DDP pre-training script', add_help=False)

    # Data parameters
    parser.add_argument('--exp_name', type=str, default="BimodalMAE_backbone")
    parser.add_argument('--output_dir', type=str, default='outputs/pretrain')
    parser.add_argument('--meta_file_path_train', type=str, 
                        default='/home/coder/workspace/data/BMI_nEMG/data/preprocessed/seg_400_meta_data_train.parquet')
    parser.add_argument('--meta_file_path_valid', type=str, 
                        default='/home/coder/workspace/data/BMI_nEMG/data/preprocessed/seg_400_meta_data_valid.parquet')
    parser.add_argument('--seq_len', type=int, default=3840, help='Sequence length')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--mask_ratio_t', type=float, default=0.5, help='Mask ratio for time domain')
    parser.add_argument('--mask_ratio_f', type=float, default=0.5, help='Mask ratio for frequency domain')

    # Architecture parameters
    parser.add_argument('--num_global_tokens', type=int, default=1, help='Number of global tokens')
    parser.add_argument('--apply_asb', type=bool, default=True, help='Apply Adaptive Spectral Block')
    parser.add_argument('--dim_tokens_enc', type=int, default=256, help='Dimension of tokens for encoder')
    parser.add_argument('--num_heads_enc', type=int, default=2, help='Number of attention heads for encoder')
    parser.add_argument('--depth_enc', type=int, default=4, help='Depth of multi-scale/multi-channel encoder')
    parser.add_argument('--dim_tokens_dec', type=int, default=192, help='Dimension of tokens for decoder')
    parser.add_argument('--num_heads_dec', type=int, default=4, help='Number of attention heads for decoder')
    parser.add_argument('--depth_dec', type=int, default=4, help='Depth of decoder')
    parser.add_argument('--mlp_ratio', type=float, default=6.0, help='MLP ratio')
    parser.add_argument('--drop_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path rate')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=0.0, help='Minimum learning rate')
    parser.add_argument('--warmup_lr', type=float, default=0.000001, help='Warmup learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='Warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--weight_decay_end', type=float, default=None, help='Final weight decay value')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume from checkpoint')

    # Optimizer parameters
    parser.add_argument('--opt', type=str, default='adamw', help='Optimizer')
    parser.add_argument('--opt_eps', type=float, default=1e-8, help='Optimizer epsilon')
    parser.add_argument('--opt_betas', type=float, nargs='+', default=[0.9, 0.95], help='Optimizer betas')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Clip gradient norm')
    parser.add_argument('--skip_grad', type=float, default=None, help='Skip update if gradient norm larger than threshold')

    # Task balancing parameters
    parser.add_argument('--task_balancer', type=str, default='none', help='Task balancing scheme')
    parser.add_argument('--balancer_lr_scale', type=float, default=1.0, help='Task loss balancer LR scale')
    parser.add_argument('--loss_weight_t', type=float, default=1.4, help='Loss weight for time domain')
    parser.add_argument('--loss_weight_f', type=float, default=0.6, help='Loss weight for frequency domain')

    # DDP parameters
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs')
    parser.add_argument('--backend', type=str, default='nccl', help='DDP backend')

    # Configuration file
    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='configs/pretrain.yaml', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    args_config, remaining = config_parser.parse_known_args()
    
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    return args


def get_model(args):
    """Creates and returns model from arguments
    """
    input_adapters = {
        'time': PatchedInputAdapter(
            num_channels=1,
            patch_size_full=args.patch_size,
            seq_len=args.seq_len
        ),
        'freq': PatchedInputAdapter(
            num_channels=1,
            patch_size_full=args.patch_size,
            seq_len=int(0.5*args.seq_len)
        )
    }

    output_adapters = {
        'time': ReConstructOutputAdapter(
            num_channels=1,
            patch_size_full=args.patch_size,
            dim_tokens_enc=args.dim_tokens_enc,
            dim_tokens_dec=args.dim_tokens_dec,
            depth=args.depth_dec,
            num_heads=args.num_heads_dec,
            seq_len=args.seq_len
        ),
        'freq': ReConstructOutputAdapter(
            num_channels=1,
            patch_size_full=args.patch_size,
            dim_tokens_enc=args.dim_tokens_enc,
            dim_tokens_dec=args.dim_tokens_dec,
            depth=args.depth_dec,
            num_heads=args.num_heads_dec,
            seq_len=int(0.5*args.seq_len)
        )
    }

    model_kwargs = {
        'num_global_tokens': args.num_global_tokens,
        'dim_tokens_enc': args.dim_tokens_enc,
        'depth_ms': args.depth_enc,
        'depth_mc': args.depth_enc,
        'num_heads': args.num_heads_enc,
        'mlp_ratio': args.mlp_ratio,
        'drop_rate': args.drop_rate,
        'attn_drop_rate': args.attn_drop_rate,
        'apply_asb': args.apply_asb
    }
    model = BimodalMAE(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        **model_kwargs
    )

    return model


def train_one_epoch(
    model: torch.nn.Module, dataloader: Iterable, 
    tasks_loss_fn: Dict[str, torch.nn.Module], loss_balancer: torch.nn.Module, optimizer: torch.optim.Optimizer,
    device: torch.device, epoch: int, global_step: int,
    loss_scaler, clip_grad: float = None, skip_grad: float = None,
    log_writer=None, lr_schedule_values=None, wd_schedule_values=None,
    num_encoded_tokens_t: int = 196, num_encoded_tokens_f: int = 12,
    loss_weight_t: float = 1.0, loss_weight_f: float = 1.0, rank: int = 0):

    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # DistributedSampler의 epoch 설정
    if hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)

    for step, inputs in enumerate(dataloader):

        # assign learning rate & weight decay for each step
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for param_group in optimizer.param_groups:
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[global_step]

        input_dict = {
            'time':inputs['time'].to(device, non_blocking=True),
            'freq':inputs['freq'].to(device, non_blocking=True),
        }

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            preds, masks = model(
                input_dict,
                num_encoded_tokens_t=num_encoded_tokens_t,
                num_encoded_tokens_f=num_encoded_tokens_f
            )

            task_losses = {
                'time':loss_weight_t * tasks_loss_fn['time'](preds['time'].float(), input_dict['time'], mask=masks['time']),
                'freq':loss_weight_f * tasks_loss_fn['freq'](preds['freq'].float(), input_dict['freq'], mask=masks['freq']),
            }
            weighted_task_losses = loss_balancer(task_losses)
            loss = sum(weighted_task_losses.values())

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, skip_grad=skip_grad,
                    parameters=model.parameters(), create_graph=is_second_order)

        lrs = [group["lr"] for group in optimizer.param_groups]
        metric = {
            'loss': sum(task_losses.values()).item(),
            'time_loss': task_losses['time'].item(),
            'freq_loss': task_losses['freq'].item(),
            'lr': max(lrs),
            'min_lr': min(lrs)
        }
        metric_logger.update(log_writer=log_writer, **metric)
        
        # 메인 프로세스에서만 로그 출력
        if rank == 0 and step % 100 == 0:
            metric_logger.log_every(step, len(dataloader), header=f'Epoch: [{epoch}]')

        global_step += 1
        if log_writer is not None:
            log_writer.set_step(global_step)
    
    return global_step


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, dataloader: Iterable, 
    tasks_loss_fn: Dict[str, torch.nn.Module], device: torch.device, 
    log_writer=None, num_encoded_tokens_t: int = 196, num_encoded_tokens_f: int = 12,
    rank: int = 0
):
    """Evaluate the model on validation dataset"""
    model.eval()
    
    total_loss = 0.0
    total_time_loss = 0.0
    total_freq_loss = 0.0
    num_samples = 0
    
    for inputs in dataloader:
        input_dict = {
            'time': inputs['time'].to(device, non_blocking=True),
            'freq': inputs['freq'].to(device, non_blocking=True),
        }
        
        with torch.amp.autocast("cuda"):
            preds, masks = model(
                input_dict,
                num_encoded_tokens_t=num_encoded_tokens_t,
                num_encoded_tokens_f=num_encoded_tokens_f
            )

            task_losses = {
                'time': tasks_loss_fn['time'](preds['time'].float(), input_dict['time'], mask=masks['time']),
                'freq': tasks_loss_fn['freq'](preds['freq'].float(), input_dict['freq'], mask=masks['freq']),
            }
            loss = sum(task_losses.values())
        
        loss_value = loss.item()
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}
        
        total_loss += loss_value
        total_time_loss += task_loss_values['time_loss']
        total_freq_loss += task_loss_values['freq_loss']
        num_samples += 1

    # DDP에서 모든 프로세스의 결과를 합산
    if dist.is_initialized():
        # 텐서로 변환하여 all_reduce 수행
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_time_loss_tensor = torch.tensor(total_time_loss, device=device)
        total_freq_loss_tensor = torch.tensor(total_freq_loss, device=device)
        num_samples_tensor = torch.tensor(num_samples, device=device)
        
        dist.all_reduce(total_loss_tensor)
        dist.all_reduce(total_time_loss_tensor)
        dist.all_reduce(total_freq_loss_tensor)
        dist.all_reduce(num_samples_tensor)
        
        total_loss = total_loss_tensor.item()
        total_time_loss = total_time_loss_tensor.item()
        total_freq_loss = total_freq_loss_tensor.item()
        num_samples = num_samples_tensor.item()

    avg_loss = total_loss / num_samples
    avg_time_loss = total_time_loss / num_samples
    avg_freq_loss = total_freq_loss / num_samples
    
    # 메인 프로세스에서만 출력
    if rank == 0:
        print(f"Validation - Loss: {avg_loss:.4f}, Time Loss: {avg_time_loss:.4f}, Freq Loss: {avg_freq_loss:.4f}")
        
        if log_writer is not None:
            log_writer.update({
                'val_loss': avg_loss,
                'val_time_loss': avg_time_loss,
                'val_freq_loss': avg_freq_loss,
            })


def run_pretrain_ddp(rank, world_size, args):
    """DDP로 실행되는 메인 학습 함수"""
    # DDP 초기화
    setup_ddp(rank, world_size, args.backend)
    
    # Dataset & DataLoader with DistributedSampler
    train_dataset = PretrainDataset(args.meta_file_path_train)
    val_dataset = PretrainDataset(args.meta_file_path_valid)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=4, pin_memory=True, drop_last=False
    )

    # Model
    device = torch.device(f'cuda:{rank}')
    model = get_model(args).to(device)
    
    # DDP로 모델 래핑
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # Task-specific Loss Criteria
    tasks_loss_fn = {
        'time': MaskedMSELoss(patch_size=args.patch_size),
        'freq': MaskedMSELoss(patch_size=args.patch_size)
    }

    if args.task_balancer == 'uncertainty':
        loss_balancer = UncertaintyWeightingStrategy(tasks=['time', 'freq'])
    else:
        loss_balancer = NoWeightingStrategy()
    loss_balancer.to(device)

    # Optimizer
    optimizer = create_optimizer(
        args, {'model': model.module, 'balancer': loss_balancer})  # model.module로 접근
    loss_scaler = NativeScaler()
    
    # Resume training 체크 및 상태 복원
    start_epoch = 0
    global_step = 0
    if args.resume and rank == 0:  # 메인 프로세스에서만 체크포인트 확인
        checkpoint_file, last_epoch = find_latest_checkpoint(args.output_dir)
        if checkpoint_file is not None:
            print(f"Resuming training from {checkpoint_file}")
            print(f"Last epoch: {last_epoch}")
            
            # 체크포인트 로드
            checkpoint = load_checkpoint(checkpoint_file)
            if checkpoint is not None:
                # 모든 프로세스에서 모델 상태 복원
                model.module.load_state_dict(checkpoint['model_state_dict'])
                
                # 옵티마이저 상태 복원
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if rank == 0:
                        print("Optimizer state restored")
                
                # loss scaler 상태 복원
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                    if rank == 0:
                        print("Loss scaler state restored")
                
                # global step 복원
                if 'global_step' in checkpoint:
                    global_step = checkpoint['global_step']
                    if rank == 0:
                        print(f"Global step restored: {global_step}")
                
                start_epoch = last_epoch + 1  # 다음 epoch부터 시작
                if rank == 0:
                    print(f"Resuming from epoch {start_epoch}")
            else:
                if rank == 0:
                    print("Failed to load checkpoint, starting from scratch")
                start_epoch = 0
        else:
            if rank == 0:
                print("No checkpoint found, starting from scratch")
            start_epoch = 0
    
    # 모든 프로세스에서 동기화
    if dist.is_initialized():
        dist.barrier()

    # LR Scheduler
    num_training_steps_per_epoch = len(train_loader)
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs
    )
    
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)

    # Mask ratio
    num_encoded_tokens_t = int(args.seq_len // args.patch_size * (1 - args.mask_ratio_t))
    num_encoded_tokens_f = int((args.seq_len//2) // args.patch_size * (1 - args.mask_ratio_f))

    # 메인 프로세스에서만 모델 정보 출력
    if rank == 0:
        print(f"Model = %s" % str(model.module))
        n_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        print(f"Number of params: {n_parameters / 1e6} M")
        print(f"Number of encoded tokens: {num_encoded_tokens_t} / {num_encoded_tokens_f}")
        print("LR = %.8f" % args.lr)
        print(f"Number of training examples per epoch = {len(train_loader.dataset)}")
        print(f"Batch size per GPU = {args.batch_size}")
        print(f"Total batch size = {args.batch_size * world_size}")
        print(f"Number of training steps per epoch = {len(train_loader)}")
        print(f"Total training steps per epoch = {num_training_steps_per_epoch}")
        print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
        print(f"World size = {world_size}")

    # 로그 라이터는 메인 프로세스에서만 생성
    log_writer = None
    if rank == 0:
        log_writer = TensorBoardLogger(args.output_dir)
    
    for epoch in range(start_epoch, args.epochs):
        global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            tasks_loss_fn=tasks_loss_fn,
            loss_balancer=loss_balancer,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            global_step=global_step,
            loss_scaler=loss_scaler,
            clip_grad=args.clip_grad,
            skip_grad=args.skip_grad,
            log_writer=log_writer,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_encoded_tokens_t=num_encoded_tokens_t,
            num_encoded_tokens_f=num_encoded_tokens_f,
            loss_weight_t=args.loss_weight_t,
            loss_weight_f=args.loss_weight_f,
            rank=rank
        )

        evaluate(
            model, val_loader, tasks_loss_fn, device, log_writer,
            num_encoded_tokens_t, num_encoded_tokens_f, rank
        )

        # 메인 프로세스에서만 체크포인트 저장
        if rank == 0 and (epoch % 10 == 0 or epoch == args.epochs-1):
            save_path = Path(args.output_dir) / 'ckpts' / f'backbone_{epoch:03d}.pth'
            save_checkpoint(save_path, epoch, model.module, optimizer, loss_scaler, global_step)

        # 모든 프로세스 동기화
        if dist.is_initialized():
            dist.barrier()

    if rank == 0 and log_writer is not None:
        log_writer.close()
    
    # DDP 정리
    cleanup_ddp()


if __name__ == '__main__':
    args = get_args()
    
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    ckpts_dir = output_dir / 'ckpts'
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    # save arguments (메인 프로세스에서만)
    args_dict = vars(args)
    yaml_path = Path(args.output_dir) / "config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(args_dict, f, allow_unicode=True, default_flow_style=False)
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.benchmark = True

    # 멀티프로세싱으로 DDP 실행
    mp.spawn(run_pretrain_ddp, args=(args.world_size, args), nprocs=args.world_size, join=True)
