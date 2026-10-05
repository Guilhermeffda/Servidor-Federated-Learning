from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from fl import runtime
from fl.model import _install_fedprox_loss


@pytest.mark.parametrize("system,hip,available,device,active", [
    ("Windows", "7.15", True, "0", True),
    ("Windows", "7.15", True, "cuda:0", True),
    ("Windows", None, True, "0", False),
    ("Linux", "7.15", True, "0", False),
    ("Windows", "7.15", False, "cpu", False),
    ("Windows", "7.15", True, "cpu", False),
    ("Darwin", None, False, "mps", False),
    ("Windows", "7.15", True, "mps", False),
])
def test_backend_selection(monkeypatch, system, hip, available, device, active):
    monkeypatch.setattr(runtime.platform, "system", lambda: system)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    original = torch.batch_norm
    enabled = torch.backends.cudnn.enabled
    with runtime.backend_compatibility(device):
        assert (torch.batch_norm is not original) == active
        assert torch.backends.cudnn.enabled == enabled
    assert torch.batch_norm is original


@pytest.mark.skipif(not runtime.is_windows_rocm(), reason="Requires physical Windows ROCm GPU")
@pytest.mark.parametrize("layer,shape", [
    (torch.nn.BatchNorm1d, (4, 3, 8)),
    (torch.nn.BatchNorm2d, (4, 3, 8, 8)),
    (torch.nn.BatchNorm3d, (2, 3, 4, 4, 4)),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_native_gpu_batchnorm_matches_documented_fallback(layer, shape, dtype):
    a = layer(3).cuda()
    b = layer(3).cuda()
    b.load_state_dict(a.state_dict())
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    z = x.detach().clone().requires_grad_()
    # Official fallback is the reference for the same native backend.
    with torch.backends.cudnn.flags(enabled=False):
        expected = a(x)
    expected.float().square().mean().backward()
    with runtime.backend_compatibility("0"):
        actual = b(z)
        actual.float().square().mean().backward()
    torch.cuda.synchronize()
    assert actual.is_cuda and z.grad.is_cuda
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(z.grad, x.grad)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert q.grad is not None and torch.isfinite(q.grad).all(), name
        torch.testing.assert_close(q.grad, p.grad)
    for name, value in a.state_dict().items():
        torch.testing.assert_close(b.state_dict()[name], value)
    a.eval()
    b.eval()
    with torch.backends.cudnn.flags(enabled=False):
        expected_eval = a(x)
    with runtime.backend_compatibility("0"):
        actual_eval = b(z)
    torch.testing.assert_close(actual_eval, expected_eval)


def test_scope_restores_nested_exception_and_isolates_threads(monkeypatch):
    monkeypatch.setattr(runtime, "is_windows_rocm", lambda: True)
    original = Mock(return_value="original")
    native = Mock(return_value=("native", None, None))
    monkeypatch.setattr(torch, "batch_norm", original)
    monkeypatch.setattr(torch, "native_batch_norm", native)
    gpu = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    cpu = SimpleNamespace(device=SimpleNamespace(type="cpu"))

    def call(input):
        return torch.batch_norm(input, None, None, None, None, True, 0.1, 1e-5, True)

    def concurrent_scope():
        with runtime.backend_compatibility("0"):
            assert call(gpu) == "native"

    with pytest.raises(RuntimeError, match="probe"):
        with runtime.backend_compatibility("0"):
            wrapper = torch.batch_norm
            assert call(gpu) == "native"
            assert call(cpu) == "original"
            with runtime.backend_compatibility("cuda:0"):
                assert torch.batch_norm is wrapper
            with ThreadPoolExecutor(max_workers=1) as executor:
                assert executor.submit(call, gpu).result() == "original"
                executor.submit(concurrent_scope).result()
            assert torch.batch_norm is wrapper
            raise RuntimeError("probe")
    assert torch.batch_norm is original
    assert runtime._users == 0


def test_cpu_batchnorm_preserves_running_stats_and_gradients(monkeypatch):
    monkeypatch.setattr(runtime, "is_windows_rocm", lambda: True)
    a = torch.nn.BatchNorm2d(3)
    b = torch.nn.BatchNorm2d(3)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 3, 4, 4, requires_grad=True)
    z = x.detach().clone().requires_grad_()
    expected = a(x)
    expected.square().mean().backward()
    with runtime.backend_compatibility("0"):
        actual = b(z)
        actual.square().mean().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(z.grad, x.grad)
    for name, value in a.state_dict().items():
        torch.testing.assert_close(b.state_dict()[name], value)


@pytest.mark.parametrize("components", [1, 3])
def test_fedprox_loss_backward_and_frozen_reference(monkeypatch, components):
    monkeypatch.setattr(runtime, "is_windows_rocm", lambda: True)
    model = torch.nn.Linear(2, 1, bias=False)
    reference = {name: p.detach().clone() for name, p in model.named_parameters()}
    snapshot = {name: p.clone() for name, p in reference.items()}
    with torch.no_grad():
        model.weight.add_(1)
    model.loss = lambda batch, preds=None: (
        (model.weight.sum() * 0).expand(components) if components > 1
        else model.weight.sum() * 0, torch.zeros(3))
    trainer = SimpleNamespace(model=model)
    with runtime.backend_compatibility("0"):
        _install_fedprox_loss(trainer, reference, 0.2)
        loss, items = model.loss({})
        loss.backward()
    torch.testing.assert_close(loss, torch.tensor(0.2))
    torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 0.2))
    assert items.shape == (3,)
    for name, value in reference.items():
        assert not value.requires_grad and value.grad is None
        torch.testing.assert_close(value, snapshot[name])


@pytest.mark.parametrize("enabled", [False, True])
def test_fixed_batch_guard_and_callback_cleanup(monkeypatch, enabled):
    monkeypatch.setattr(runtime, "is_windows_rocm", lambda: enabled)
    class YOLOProbe:
        def __init__(self):
            self.callbacks = {"on_train_batch_start": [], "on_pretrain_routine_end": []}

        def add_callback(self, event, callback):
            self.callbacks[event].append(callback)

        def train(self, **kwargs):
            trainer = SimpleNamespace(_oom_retries=0)
            for callback in self.callbacks["on_train_batch_start"]:
                callback(trainer)
            assert trainer._oom_retries == (3 if enabled else 0)
            assert kwargs["batch"] == 16
            raise RuntimeError("out of memory")

    model = YOLOProbe()
    original = torch.batch_norm
    with pytest.raises(RuntimeError, match="out of memory"):
        runtime.run_yolo(model.train, device="0", batch=16)
    assert model.callbacks["on_train_batch_start"] == []
    assert model.callbacks["on_pretrain_routine_end"] == []
    assert torch.batch_norm is original


@pytest.mark.parametrize("hardware,device,resume,enabled,initial,expected", [
    (True, "cuda:0", False, True, 65536., 1024.),
    (True, "cuda:0", False, True, 128., 128.),
    (True, "cuda:0", True, True, 65536., 65536.),
    (True, "cpu", False, True, 65536., 65536.),
    (False, "cuda:0", False, True, 65536., 65536.),
    (True, "cuda:0", False, False, 65536., 65536.),
])
def test_amp_scale_selection(monkeypatch, hardware, device, resume, enabled, initial, expected):
    monkeypatch.setattr(runtime, "is_windows_rocm", lambda: hardware)
    state = {"scale": initial, "growth_factor": 2.0, "backoff_factor": 0.5,
             "growth_interval": 2000, "_growth_tracker": 0}
    scaler = SimpleNamespace(is_enabled=lambda: enabled, state_dict=lambda: dict(state),
                             load_state_dict=lambda received: state.update(received))
    runtime.configure_amp_scaler(SimpleNamespace(scaler=scaler, device=device, resume=resume))
    assert state["scale"] == expected
    assert state["growth_factor"] == 2.0 and state["growth_interval"] == 2000
