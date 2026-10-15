"""Hardware selection independent of GPU product name."""
import re


class Hardware:
    def __init__(self, torch, requested='auto', precision='auto'):
        if requested not in ('auto', 'cpu', 'cuda') and not re.fullmatch(r'cuda:\d+', requested):
            raise ValueError('Device must be auto, cpu, cuda, or cuda:N')
        if precision not in ('auto', 'fp32'):
            raise ValueError('Precision must be auto or fp32')
        self.torch = torch
        use_cuda = requested != 'cpu' and torch.cuda.is_available()
        if requested.startswith('cuda') and not use_cuda:
            raise RuntimeError('CUDA requested but unavailable. Install a compatible NVIDIA driver and CUDA-enabled PyTorch, or choose cpu.')
        self.index = int(requested.split(':')[1]) if ':' in requested else 0
        if use_cuda and self.index >= torch.cuda.device_count():
            raise ValueError(f'GPU {self.index} is unavailable; detected {torch.cuda.device_count()} GPU(s)')
        self.cuda = use_cuda
        self.device = f'cuda:{self.index}' if use_cuda else 'cpu'
        self.predict_device = self.index if use_cuda else 'cpu'
        self.fp16 = use_cuda and precision == 'auto' and torch.cuda.get_device_capability(self.index) >= (5, 3)
        self.name = torch.cuda.get_device_name(self.index) if use_cuda else 'CPU (slower processing)'
        self.activate()
        try:
            # Check actual execution, not just GPU enumeration, before loading the pipeline.
            probe = torch.ones((16, 16), device=self.device, dtype=torch.float16 if self.fp16 else torch.float32)
            _ = probe @ probe
            self.synchronize()
        except RuntimeError as exc:
            raise RuntimeError(f'{self.name} execution check failed. Check the driver/PyTorch compatibility; try --precision fp32 if half precision failed. {exc}') from exc

    def activate(self):
        if self.cuda:
            self.torch.cuda.set_device(self.index)

    def synchronize(self):
        if self.cuda:
            self.torch.cuda.synchronize(self.index)

    def report(self):
        return dict(device=self.device, name=self.name, precision='FP16' if self.fp16 else 'FP32',
                    torch=self.torch.__version__, cuda_runtime=self.torch.version.cuda)


if __name__ == '__main__':
    import argparse
    import json
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto')
    parser.add_argument('--precision', choices=['auto', 'fp32'], default='auto')
    args = parser.parse_args()
    print(json.dumps(Hardware(torch, args.device, args.precision).report(), indent=2))
