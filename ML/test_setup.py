"""Portable installation choices must not depend on the original laptop."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('setup_aura', Path(__file__).resolve().parents[1] / 'setup_aura.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


@pytest.mark.parametrize('device,channel,names,expected', [
    ('auto', 'auto', 'NVIDIA GeForce RTX 4050 Laptop GPU', ('cu121', '2.5.1', '0.20.1')),
    ('auto', 'auto', 'NVIDIA GeForce RTX 2050', ('cu121', '2.5.1', '0.20.1')),
    ('auto', 'auto', 'NVIDIA GeForce RTX 5070', ('cu128', '2.7.1', '0.22.1')),
    ('auto', 'auto', '', ('cpu', '2.5.1', '0.20.1')),
    ('cpu', 'auto', 'NVIDIA GPU', ('cpu', '2.5.1', '0.20.1')),
    ('cuda', 'cu128', '', ('cu128', '2.7.1', '0.22.1')),
])
def test_choose_runtime(device, channel, names, expected):
    assert setup.choose_runtime(device, channel, names) == expected


def test_cpu_rejects_cuda_channel():
    with pytest.raises(ValueError):
        setup.choose_runtime('cpu', 'cu121', '')
