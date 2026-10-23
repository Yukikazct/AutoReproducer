"""Platform-specific versions for the project's fixed author-code runtimes.

PyTorch/PyG publish macOS CPU wheels without the local-version suffixes used
by their Windows/Linux CPU builds. Keep planning, installation and runtime
repairs on the same map instead of rewriting requirements at pip invocation.
"""
import sys


CPU_COMPATIBILITY_PINS = {
    "torch": "2.5.1+cpu", "torchvision": "0.20.1+cpu", "numpy": "1.26.4",
    "scipy": "1.14.1", "pandas": "2.2.3", "scikit-learn": "1.5.2",
    "matplotlib": "3.9.2", "pillow": "10.4.0", "networkx": "2.8.8",
    "tqdm": "4.67.1", "hyperopt": "0.2.7", "seaborn": "0.13.2", "tensorboardx": "2.6.2.2",
    "googledrivedownloader": "0.4", "requests": "2.32.3",
    "gdown": "5.2.0",
    "torch-geometric": "2.6.1", "pyyaml": "6.0.2", "absl-py": "2.1.0", "tensorboard": "2.18.0",
    "torch-scatter": "2.1.2+pt25cpu",
    "torch-sparse": "0.6.18+pt25cpu", "dgl": "1.1.2", "setuptools": "69.5.1", "packaging": "24.2",
    "aiohttp": "3.10.10", "fsspec": "2024.10.0", "jinja2": "3.1.6",
    "psutil": "6.1.0", "pyparsing": "3.2.0",
    "optuna": "3.6.1", "deeprobust": "0.2.11", "ogb": "1.3.6",
    "nbclient": "0.10.2", "nbformat": "5.10.4", "ipykernel": "6.29.5",
}
CUDA_COMPATIBILITY_PINS = {
    **{name: version for name, version in CPU_COMPATIBILITY_PINS.items() if name != "dgl"},
    "torch": "2.5.1+cu121", "torchvision": "0.20.1+cu121",
    "torch-scatter": "2.1.2+pt25cu121", "torch-sparse": "0.6.18+pt25cu121",
}
_MACOS_CPU_PINS = {
    "torch": "2.5.1", "torchvision": "0.20.1",
    "torch-scatter": "2.1.2", "torch-sparse": "0.6.18",
}


def compatibility_pins(device="cpu", *, platform_name=None):
    """Return a fresh version map for the interpreter executing author code.

    A requested CUDA experiment must retain that requirement; macOS cannot
    silently substitute CPU or MPS and still run the same experiment.
    """
    host = sys.platform if platform_name is None else platform_name
    if device not in {"cpu", "cuda"}:
        raise ValueError("作者执行设备必须为 cpu 或 cuda")
    if device == "cuda":
        if host == "darwin":
            raise ValueError("macOS 不支持本项目的 NVIDIA CUDA 运行环境；请选择 CPU 实验或使用支持 CUDA 的设备。")
        return dict(CUDA_COMPATIBILITY_PINS)
    pins = dict(CPU_COMPATIBILITY_PINS)
    if host == "darwin":
        pins.update(_MACOS_CPU_PINS)
    return pins


def reviewed_python_requirement(requirements, device="cpu", *, platform_name=None):
    """State the tested native combination, separately from upstream metadata."""
    host = sys.platform if platform_name is None else platform_name
    if (host in {"win32", "linux"} and device == "cpu"
            and any(line.strip().lower() == "dgl==1.1.2" for line in requirements)):
        return {"required_python": "==3.11.*",
                "required_python_source": "reviewed_native_cpu:dgl==1.1.2"}
    return {}
