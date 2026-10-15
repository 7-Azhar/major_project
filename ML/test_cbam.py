"""CPU smoke test of trainer reconstruction, gradient flow, and checkpoint reload."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import torch
from ultralytics import YOLO
from aura_cbam import CBAM, CBAMTrainer, cbam_count, verify_training_model


def test_trainer_retains_cbam():
    torch.set_num_threads(2)
    source = YOLO(str(Path(__file__).parent / 'models/yolov8m_visdrone_recovered.pt'))
    trainer = CBAMTrainer.__new__(CBAMTrainer)
    trainer.args = SimpleNamespace(cls_remap=True)
    trainer.data = {'nc': 10, 'channels': 3, 'names': source.names}
    model = trainer.get_model(cfg=source.model.yaml, weights=source.model, verbose=False)
    assert cbam_count(model) == 3
    # Original convolution weights must survive initialization from the baseline.
    assert torch.equal(model.model[2][0].cv1.conv.weight, source.model.model[2].cv1.conv.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    verify_training_model(SimpleNamespace(model=model, optimizer=optimizer))
    model.eval()
    output = model(torch.rand(1, 3, 128, 128))[0]
    output.square().mean().backward()
    for module in model.modules():
        if isinstance(module, CBAM):
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
            assert any(p.grad.abs().sum() > 0 for p in module.parameters())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'cbam_smoke.pt'
        source.model = model
        source.save(path)
        reloaded = YOLO(str(path))
        assert cbam_count(reloaded.model) == 3
        with torch.no_grad():
            prediction = reloaded.model.float().eval()(torch.rand(1, 3, 128, 128))[0]
            assert torch.isfinite(prediction).all()


if __name__ == '__main__':
    test_trainer_retains_cbam()
    print('PASS: trainer insertion, pretrained weights, optimizer inclusion, gradients, checkpoint reload')
