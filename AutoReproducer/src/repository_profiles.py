"""Reviewed, fixed experiments; these plans never come from generated shell text."""
from copy import deepcopy
from src.method_profiles import METHOD_LABELS, method_profile

PAPER_TITLE = "Are Transformers Effective for Time Series Forecasting?"
REPO_URL = "https://github.com/cure-lab/LTSF-Linear"
REPO_SHA = "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
DATA_REVISION = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
DATA_URL = (f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{DATA_REVISION}"
            "/ETT-small/ETTh1.csv")
DATA_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"

PROFILE_LABELS = {
    "dlinear_etth1_reference": "DLinear · ETTh1 · 完整实验（作者训练协议）",
    "dlinear_etth1_smoke": "DLinear · ETTh1 · 快速诊断（1轮）",
}
PROFILE_LABELS.update(METHOD_LABELS)


def get_profile(profile_id):
    if profile_id in METHOD_LABELS:
        return method_profile(profile_id)
    if profile_id not in PROFILE_LABELS:
        raise ValueError(f"未适配的论文预设: {profile_id}")
    smoke = profile_id == "dlinear_etth1_smoke"
    params = {"seq_len": 96 if smoke else 336, "pred_len": 96,
              "train_epochs": 1 if smoke else 10, "batch_size": 32,
              "learning_rate": 0.005, "patience": 3, "num_workers": 0,
              "itr": 1, "seed": 2021, "features": "M", "enc_in": 7,
              "device": "cpu"}
    model_id = f"ETTh1_{params['seq_len']}_96"
    argv = ["python", "-u", "run_longExp.py", "--is_training", "1",
            "--model_id", model_id, "--model", "DLinear", "--data", "ETTh1",
            "--root_path", "./dataset/", "--data_path", "ETTh1.csv",
            "--features", "M", "--seq_len", str(params["seq_len"]),
            "--pred_len", "96", "--enc_in", "7", "--des", "Exp",
            "--itr", "1", "--batch_size", "32", "--learning_rate", "0.005",
            "--train_epochs", str(params["train_epochs"]), "--patience", "3",
            "--num_workers", "0"]
    return deepcopy({
        "version": 1, "id": profile_id, "label": PROFILE_LABELS[profile_id],
        "paper": {"title": PAPER_TITLE, "url": "https://arxiv.org/abs/2205.13504",
                  "method": "DLinear", "dataset": "ETTh1",
                  "metrics": {} if smoke else {"mse": 0.375, "mae": 0.399},
                  "required_metrics": ["mse", "mae"],
                  "reference_source": "https://arxiv.org/html/2205.13504v3#S5.T2"},
        "repository": {"url": REPO_URL, "revision": REPO_SHA},
        "dataset": {"name": "ETTh1", "url": DATA_URL, "revision": DATA_REVISION,
                    "sha256": DATA_SHA256, "bytes": 2589657, "rows": 17420,
                    "target": "dataset/ETTh1.csv", "kind": "real",
                    "split": "官方时间切分：训练前8640行，验证至11520行，测试至14400行；scaler只拟合训练集"},
        "parameters": params,
        "environment": {
            "requirements_txt": "numpy==1.26.4\npandas==2.2.3\nscikit-learn==1.5.2\nmatplotlib==3.9.2\ntorch==2.5.1\n",
            "note": "Python 3.11/3.12 CPU兼容环境；原作者torch==1.9.0，算法源码保持固定版本。首次运行会安装隔离依赖。",
        },
        "steps": [
            {"id": "import_check", "kind": "check", "cwd": ".", "timeout_s": 90,
             "depends_on": [],
             "argv": ["python", "-c", "import sys,platform,torch,numpy,pandas,sklearn,matplotlib; from exp.exp_main import Exp_Main; import models.DLinear; print({'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':numpy.__version__,'pandas':pandas.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__})"]},
            {"id": "train_and_eval", "kind": "train", "argv": argv, "cwd": ".",
             "timeout_s": 600 if smoke else 1800,
             "depends_on": ["import_check"],
             # Author output directory names contain the run timestamp, hence the glob.
             "artifacts": [{"path": "results/*/checkpoint.pth"}, {"path": "results/*/pred.npy"}],
             "env": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                     "MKL_NUM_THREADS": "2", "MPLBACKEND": "Agg", "PYTHONHASHSEED": "2021"}},
        ],
        "validation": {"level": "smoke" if smoke else "reference",
                       "relative_tolerance": 0.05,
                       "protocol_verified": False,
                       "scope": "selected_paper_experiment",
                       "reference_aggregation": "single_seed",
                       "aggregation_source": "https://github.com/cure-lab/LTSF-Linear/issues/33#issuecomment-1331937601",
                       "implementation_note": "使用固定作者仓库0c113版本；作者后续模型初始化与早期表格版本有差异，记录版本并按事前5%容差核对。",
                       "note": "范围为Table 2的DLinear/ETTh1多变量336→96单项实验。作者确认论文只使用单seed；完整训练按官方早停并加载验证集最优checkpoint。仅在运行协议与独立指标复算通过后判定本项数值复现。"},
        "repository_map": {"model": "models/DLinear.py", "data_split": "data_provider/data_loader.py",
                           "training_and_test": "exp/exp_main.py", "metrics": "utils/metrics.py",
                           "author_command": "scripts/EXP-LongForecasting/Linear/etth1.sh"},
    })
