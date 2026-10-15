from unittest.mock import MagicMock
import pytest
from hardware import Hardware


def fake_torch(available=True, count=2):
    t = MagicMock()
    t.cuda.is_available.return_value = available
    t.cuda.device_count.return_value = count
    t.cuda.get_device_capability.return_value = (8, 9)
    t.cuda.get_device_name.return_value = 'Test RTX'
    return t


def test_auto_uses_cuda_and_half_precision():
    t = fake_torch()
    h = Hardware(t)
    assert h.device == 'cuda:0' and h.fp16
    assert t.ones.call_args.kwargs['device'] == 'cuda:0'


def test_explicit_second_gpu_and_fp32():
    t = fake_torch()
    h = Hardware(t, 'cuda:1', 'fp32')
    assert h.predict_device == 1 and not h.fp16
    t.cuda.set_device.assert_called_with(1)
    t.cuda.synchronize.assert_called_with(1)


def test_cpu_fallback_never_calls_cuda_execution():
    t = fake_torch(False)
    h = Hardware(t)
    assert h.device == 'cpu' and not h.fp16
    t.cuda.set_device.assert_not_called()
    t.cuda.synchronize.assert_not_called()


def test_requested_cuda_does_not_silently_fallback():
    with pytest.raises(RuntimeError, match='CUDA requested'):
        Hardware(fake_torch(False), 'cuda')


def test_invalid_gpu_index_rejected():
    with pytest.raises(ValueError, match='unavailable'):
        Hardware(fake_torch(count=1), 'cuda:1')


def test_incompatible_runtime_reported():
    t = fake_torch()
    t.ones.side_effect = RuntimeError('no kernel image')
    with pytest.raises(RuntimeError, match='driver/PyTorch'):
        Hardware(t)
