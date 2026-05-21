from pathlib import Path
from collections import defaultdict, deque
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

class SimpleValue:
    def __init__(self, fmt='{value:.6f}'):
        self.fmt = fmt
        self.value = 0.0
    
    def update(self, value, n=1):
        self.value = value
    
    def __str__(self):
        return self.fmt.format(value=self.value)

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / (self.count + 1e-8)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, log_writer=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
        
        if log_writer is not None:
            log_writer.update(kwargs)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, step, num_steps, header=None):
        if not header:
            header = ''
        space_fmt = ':' + str(len(str(num_steps))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            '{meters}'
        ]
        log_msg = self.delimiter.join(log_msg)
        print(log_msg.format(step, num_steps, meters=str(self)))


class TensorBoardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, metrics):
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, dict):
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.writer.add_scalar(k, v, self.step)

    def close(self):
        self.writer.close()

def _fmt_ci(value, ci_tuple):
    """Format metric with 95% CI: 0.8234 (0.7891 - 0.8576)"""
    if ci_tuple is None or (isinstance(ci_tuple, tuple) and any(v != v for v in ci_tuple)):
        return f"{value:.4f}"
    lo, hi = ci_tuple
    return f"{value:.4f} ({lo:.4f} - {hi:.4f})"


def save_report(output_dir, metrics):
    save_path = Path(output_dir) / "eval_result.txt"
    with open(save_path, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*50 + "\n")
        class_names = ['nl', 'n', 'm']

        f.write(f"Accuracy: {metrics['acc']:.4f}\n")
        f.write(f"AUROC: {_fmt_ci(metrics['auc'], metrics.get('auroc_ci'))}\n")
        roc_aucs = metrics.get('roc_aucs')
        auroc_ci_per = metrics.get('auroc_ci_per_class')
        if roc_aucs is not None:
            for i, (name, val) in enumerate(zip(class_names, roc_aucs)):
                ci = auroc_ci_per[i] if (auroc_ci_per and i < len(auroc_ci_per)) else None
                f.write(f"  AUROC [{name}]: {_fmt_ci(val, ci)}\n")
        f.write(f"AUPRC: {_fmt_ci(metrics.get('auprc', 0), metrics.get('auprc_ci'))}\n")
        auprc_per = metrics.get('auprc_per_class')
        auprc_ci_per = metrics.get('auprc_ci_per_class')
        if auprc_per is not None:
            for i, (name, val) in enumerate(zip(class_names, auprc_per)):
                ci = auprc_ci_per[i] if (auprc_ci_per and i < len(auprc_ci_per)) else None
                f.write(f"  AUPRC [{name}]: {_fmt_ci(val, ci)}\n")
        f.write(f"macro-F1: {_fmt_ci(metrics['f1'], metrics.get('macro_f1_ci'))}\n")
        f.write(f"weighted-F1: {_fmt_ci(metrics.get('f1_weighted', 0), metrics.get('weighted_f1_ci'))}\n")
        f1_per = metrics.get('f1_per_class')
        f1_ci_per = metrics.get('f1_ci_per_class')
        if f1_per is not None:
            for i, (name, val) in enumerate(zip(class_names, f1_per)):
                ci = f1_ci_per[i] if (f1_ci_per and i < len(f1_ci_per)) else None
                f.write(f"  F1 [{name}]: {_fmt_ci(val, ci)}\n")

        f.write("\nDETAILED CLASSIFICATION REPORT\n")
        f.write("-"*30 + "\n")
        f.write(str(metrics['classification_report']))

        f.write("\nCONFUSION MATRIX\n")
        f.write("-"*30 + "\n")
        f.write(str(metrics['confusion_matrix']))
        f.write("\n" + "="*50 + "\n")

    print(f"\nResults saved to: {save_path}")

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
