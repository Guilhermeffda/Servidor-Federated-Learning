"""Hardware detection and scoped Windows ROCm BatchNorm compatibility.

Only eager BatchNorm dispatch is redirected, to ATen native BatchNorm on the
same GPU. Functional/module validation, running statistics and autograd remain
PyTorch's. No backend flag, model architecture or state_dict is changed.
"""

from __future__ import annotations

import platform
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from threading import RLock

import torch

_active = ContextVar("windows_rocm_batchnorm", default=False)
_lock = RLock()
_users = 0
_original = None
WINDOWS_ROCM_AMP_INIT_SCALE = 1024.0


def configure_amp_scaler(trainer):
    """Avoid losing every short local update to initial FP16 scale overflow.

    This is numerical loss scaling, unscaled before optimizer.step; AMP,
    accumulation, LR and the objective are unchanged. Keep restored checkpoint
    scales on resume, disabled scalers, and other hardware untouched.
    """
    scaler = getattr(trainer, "scaler", None)
    if (not is_windows_rocm() or str(getattr(trainer, "device", "cpu")).split(":")[0] != "cuda"
            or getattr(trainer, "resume", False) or scaler is None or not scaler.is_enabled()):
        return
    state = scaler.state_dict()
    state["scale"] = min(state["scale"], WINDOWS_ROCM_AMP_INIT_SCALE)
    scaler.load_state_dict(state)


def run_yolo(method, **kwargs):
    """Run an Ultralytics train/val call with scoped backend compatibility."""
    owner = getattr(method, "__self__", None)
    guarded = (is_windows_rocm() and uses_cuda_api(kwargs.get("device"))
               and getattr(method, "__name__", "") == "train" and owner is not None)
    # Recent Ultralytics versions silently halve a fixed batch after an OOM.
    # Exhaust its retry budget before each batch: propagate the original error
    # instead. Older versions do not consume this attribute. No change on CUDA.
    def fixed_batch(trainer):
        trainer._oom_retries = 3

    if guarded:
        if kwargs.get("batch", 1) <= 0:
            raise ValueError("Windows ROCm scientific training requires a fixed positive batch")
        owner.add_callback("on_train_batch_start", fixed_batch)
        owner.add_callback("on_pretrain_routine_end", configure_amp_scaler)
    try:
        with backend_compatibility(kwargs.get("device")):
            return method(**kwargs)
    finally:
        if guarded:
            owner.callbacks["on_train_batch_start"].remove(fixed_batch)
            owner.callbacks["on_pretrain_routine_end"].remove(configure_amp_scaler)


def is_rocm() -> bool:
    return torch.version.hip is not None


def is_windows_rocm() -> bool:
    return platform.system() == "Windows" and is_rocm() and torch.cuda.is_available()


def uses_cuda_api(device: str | torch.device | None) -> bool:
    if device is None:
        return True  # Ultralytics auto-selection/resume; input is checked too.
    value = str(device).lower()
    return value.startswith("cuda") or all(part.strip().isdigit() for part in value.split(","))


def get_accelerator_info(device: str | torch.device | None = None) -> dict:
    available = torch.cuda.is_available()
    selected = str(device) if device is not None else ("0" if available else "cpu")
    gpu = available and uses_cuda_api(selected)
    index = int(selected.split(",")[0]) if selected.split(",")[0].isdigit() else 0
    if selected.startswith("cuda:"):
        index = int(selected.split(":")[1])
    return {
        "system": platform.system(), "platform": platform.platform(),
        "python": platform.python_version(), "torch_version": torch.__version__,
        "backend": ("rocm" if is_rocm() else "cuda") if gpu else ("mps" if selected == "mps" else "cpu"),
        "cuda_version": torch.version.cuda, "hip_version": torch.version.hip,
        "cuda_available": available, "device": selected,
        "gpu_name": torch.cuda.get_device_name(index) if gpu else None,
        "vram_bytes": torch.cuda.get_device_properties(index).total_memory if gpu else None,
        "batchnorm_native": is_windows_rocm() and uses_cuda_api(selected),
        "amp_initial_scale_limit": WINDOWS_ROCM_AMP_INIT_SCALE if is_windows_rocm() and uses_cuda_api(selected) else None,
    }


@contextmanager
def backend_compatibility(device: str | torch.device | None = None):
    """Native BatchNorm only for GPU calls in this thread's ROCm scope.

    A reference-counted dispatch wrapper supports nested/concurrent contexts;
    ContextVar makes other threads and CPU/MPS calls delegate unchanged. The
    installation lock is never held during training. Exceptions restore both
    context and the original callable after the last scope exits. Do not replace
    torch.batch_norm independently while this context is active. Compiled/JIT
    models and distributed SyncBatchNorm are outside this eager YOLO pipeline.
    """
    global _users, _original
    if not (is_windows_rocm() and uses_cuda_api(device)):
        yield
        return
    with _lock:
        if _users == 0:
            original = torch.batch_norm
            _original = original

            @wraps(original)
            def batch_norm(input, weight, bias, running_mean, running_var,
                           training, momentum, eps, cudnn_enabled):
                if _active.get() and input.device.type == "cuda":
                    return torch.native_batch_norm(
                        input, weight, bias, running_mean, running_var,
                        training, momentum, eps,
                    )[0]
                return original(input, weight, bias, running_mean, running_var,
                                training, momentum, eps, cudnn_enabled)

            torch.batch_norm = batch_norm
        _users += 1
    token = _active.set(True)
    try:
        yield
    finally:
        _active.reset(token)
        with _lock:
            _users -= 1
            if _users == 0:
                torch.batch_norm = _original
                _original = None
