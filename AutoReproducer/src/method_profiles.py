"""Frozen official method experiments, distinct from paper-table reproduction."""
from copy import deepcopy

SIREN_SHA = "4df34baee3f0f9c8f351630992c1fe1f69114b5f"
SIREN_NOTEBOOK_SHA256 = "ac8bbdb970916bfb344d406d11d98a9746565036fae4d251dd1e67a70f7b1be6"
CAMERA_SHA256 = "361a6d56d22ee52289cd308d5461d090e06a56cb36007d8dfc3226cbe8aaa5db"
ODE_SHA = "657943acefa826ef04c025ebeb1ff5e9d60dc268"
METHOD_LABELS = {"siren_camera_quick": "SIREN · 图像拟合 · 五分钟快速档（环境自动准备）",
                 "neural_ode_spiral": "Neural ODE · 官方螺旋轨迹拟合"}
COMMON_REQUIREMENTS = "numpy==1.26.4\nscipy==1.14.1\nmatplotlib==3.9.2\nPillow==10.4.0\n"


def method_profile(profile_id):
    if profile_id not in METHOD_LABELS:
        raise ValueError(f"未适配的论文预设: {profile_id}")
    if profile_id == "neural_ode_spiral":
        return deepcopy({
            "version": 1, "id": profile_id, "adapter_id": "neural_ode", "label": METHOD_LABELS[profile_id],
            "paper": {"title": "Neural Ordinary Differential Equations", "url": "https://arxiv.org/abs/1806.07366",
                      "method": "Neural ODE", "dataset": "Official cubic spiral ODE", "metrics": {},
                      "required_metrics": ["mae", "rmse"],
                      "reference_source": f"https://github.com/rtqichen/torchdiffeq/blob/{ODE_SHA}/examples/ode_demo.py"},
            "repository": {"url": "https://github.com/rtqichen/torchdiffeq", "revision": ODE_SHA},
            "source_sha256": "36cc0ebfcf56afb7a291142c03a19b8f0bd3d437f914f3f98657e048aa8a60e4",
            "required_files": ["examples/ode_demo.py", "torchdiffeq/__init__.py", "README.md"],
            "dataset": {"name": "cubic-spiral", "kind": "analytic", "url": f"https://github.com/rtqichen/torchdiffeq/blob/{ODE_SHA}/examples/ode_demo.py",
                        "equation": "dy/dt = (y**3) @ [[-0.1,2.0],[-2.0,-0.1]]", "initial_state": [2., 0.],
                        "time_range": [0., 25.], "size": 1000,
                        "split": "官方全轨迹采样训练；新增初值的验证和留出评估仅用于优化扩展"},
            "parameters": {"seed": 2021, "steps": 2000, "data_size": 1000, "batch_time": 10,
                           "batch_size": 20, "learning_rate": .001, "device": "cpu", "solver": "dopri5",
                           "rtol": 1e-7, "atol": 1e-9, "protocol": "official_fit"},
            "environment": {"requirements_txt": "--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.5.1+cpu\n" + COMMON_REQUIREMENTS,
                            "note": "固定 torchdiffeq 作者源码，Python 3.11/3.12 与 PyTorch 2.5.1 CPU 兼容环境。"},
            "validation": {"level": "method", "scope": "official_method_experiment",
                           "note": "官方螺旋拟合方法实验；MAE 为轨迹拟合误差，不是论文完整基准。独立参考轨迹使用高精度 SciPy DOP853。"},
            "budget": {"total_s": 1260, "baseline_s": 1200, "advice_s": 45},
            "search_space": {"learning_rate": [.0003, .001, .003], "batch_time": [10, 20, 40]},
        })
    return deepcopy({
        "version": 1, "id": profile_id, "adapter_id": "siren",
        "label": METHOD_LABELS[profile_id],
        "paper": {"title": "Implicit Neural Representations with Periodic Activation Functions",
                  "url": "https://arxiv.org/abs/2006.09661", "method": "SIREN",
                  "dataset": "cameraman (scikit-image 0.16.2)", "metrics": {},
                  "required_metrics": ["mse", "psnr"],
                  "reference_source": f"https://github.com/vsitzmann/siren/blob/{SIREN_SHA}/explore_siren.ipynb"},
        "repository": {"url": "https://github.com/vsitzmann/siren", "revision": SIREN_SHA},
        "source_sha256": SIREN_NOTEBOOK_SHA256,
        "required_files": ["explore_siren.ipynb", "README.md", "LICENSE"],
        "dataset": {"name": "camera-0.16.2", "url": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.16.2/skimage/data/camera.png",
                    "sha256": CAMERA_SHA256, "bytes": 114228, "target": "camera.png",
                    "kind": "real", "split": "官方全图拟合；不是留出测试"},
        "parameters": {"seed": 2021, "steps": 500, "sidelength": 256, "hidden_features": 256,
                       "hidden_layers": 3, "first_omega_0": 30, "hidden_omega_0": 30,
                       "learning_rate": 0.0001, "device": "cuda", "protocol": "official_fit"},
        "environment": {"requirements_txt": "--extra-index-url https://download.pytorch.org/whl/cu121\n"
                        "torch==2.5.1+cu121\ntorchvision==0.20.1+cu121\n" + COMMON_REQUIREMENTS,
                        "note": "Python 3.11/3.12；独立 CUDA 12.1 依赖缓存；先准备环境再运行。"},
        "validation": {"level": "method", "scope": "official_method_experiment",
                       "min_psnr_db": 25.0, "note": "完整官方图像拟合示例；25 dB 是项目工程门槛，不是论文公布阈值。"},
        "budget": {"total_s": 300, "baseline_s": 240, "advice_s": 45},
        "search_space": {"learning_rate": [0.00005, 0.0001, 0.0002], "first_omega_0": [15, 30, 60]},
    })
