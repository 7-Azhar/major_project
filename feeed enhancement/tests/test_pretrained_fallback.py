"""Regression: reduced-curve fallbacks must never silently use random weights."""
from pathlib import Path
import pytest


def test_four_curve_fallback_loads_pretrained_weights(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    from app.enhancers.neural import ZeroDceEnhancer
    root = Path(__file__).resolve().parents[1]
    checkpoint = torch.load(root / 'outputs/models/Epoch99.pth', map_location='cpu', weights_only=True)
    monkeypatch.chdir(tmp_path)
    enhancer = ZeroDceEnhancer(device='cpu', fp16=False, num_curves=4)
    assert torch.equal(enhancer.net.conv1.weight, checkpoint['e_conv1.weight'])
    assert torch.equal(enhancer.net.conv7.weight, checkpoint['e_conv7.weight'][:12])


def test_missing_checkpoint_fails_explicitly(tmp_path):
    pytest.importorskip('torch')
    from app.enhancers.neural import ZeroDceEnhancer
    with pytest.raises(FileNotFoundError, match='Pretrained'):
        ZeroDceEnhancer(model_path=str(tmp_path / 'missing.pth'), device='cpu')
