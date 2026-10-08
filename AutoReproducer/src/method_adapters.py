"""Deterministic preparation and validation for reviewed method experiments."""
import hashlib
import ast
import json
import math
import shutil
import urllib.request
import uuid
from pathlib import Path

from src.method_profiles import SIREN_NOTEBOOK_SHA256
from src.safety.paths import workspace_path

RUNTIMES = Path(__file__).parent / "experiments"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def download_dataset(root, spec, workspace, offline=False):
    cache = Path(root) / "dataset_cache" / spec["name"] / spec["sha256"] / spec["target"]
    if not cache.is_file() or digest(cache) != spec["sha256"]:
        if offline:
            raise RuntimeError("离线缓存缺少固定数据或校验值不符；请先准备实验环境")
        with urllib.request.urlopen(spec["url"], timeout=30) as response:
            content = response.read(spec["bytes"] + 1)
        if len(content) != spec["bytes"] or hashlib.sha256(content).hexdigest() != spec["sha256"]:
            raise ValueError("数据下载内容校验失败")
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_name(f".{uuid.uuid4().hex}.part")
        temp.write_bytes(content); temp.replace(cache)
    target = workspace_path(workspace, spec["target"], "dataset")
    shutil.copyfile(cache, target)
    return {**spec, "path": str(target.resolve()), "verified": True}


class SirenAdapter:
    runtime = "siren_runtime.py"
    evaluator = "evaluate_siren.py"

    def required_files(self, profile):
        return profile["required_files"]

    def prepare_dataset(self, root, profile, workspace, *, offline=False):
        return download_dataset(root, profile["dataset"], workspace, offline)

    def public_sources(self, workspace, profile):
        notebook = read_json(Path(workspace) / "explore_siren.ipynb")
        url = profile["paper"]["reference_source"]
        return [{"source_id": f"author_cell_{i}", "url": url, "locator": f"cell[{i}]",
                 "content": "".join(notebook["cells"][i]["source"])} for i in (3, 11, 13)]

    def materialize(self, workspace, profile, spec_hash):
        root = Path(workspace)
        # git archive honors the local export EOL policy on Windows. The fixed
        # source checksum is LF-normalized; snapshot hashes retain actual bytes.
        notebook_bytes = (root / "explore_siren.ipynb").read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(notebook_bytes).hexdigest() != SIREN_NOTEBOOK_SHA256:
            raise ValueError("固定 SIREN notebook 内容校验失败")
        model = "".join(read_json(root / "explore_siren.ipynb")["cells"][3]["source"])
        (root / "author_model.py").write_text(
            "import torch\nfrom torch import nn\nimport numpy as np\nfrom collections import OrderedDict\n\n" + model,
            encoding="utf-8")
        return self._write_runtime(root, profile, spec_hash, ["author_model.py"])

    def _write_runtime(self, root, profile, spec_hash, additional=()):
        for source, target in [(self.runtime, "run_experiment.py"), (self.evaluator, "evaluate.py")]:
            shutil.copyfile(RUNTIMES / source, root / target)
        write_json(root / "experiment.json", {"parameters": profile["parameters"],
                   "dataset_sha256": profile["dataset"].get("sha256"), "spec_sha256": spec_hash})
        files = ["run_experiment.py", "evaluate.py", "experiment.json", *additional]
        manifest = {"files": {name: digest(root / name) for name in files},
                    "source": profile["paper"]["reference_source"],
                    "note": "保留官方模型、初始化与训练算法；增加固定种子、参数与产物记录，绘图移至独立评估。"}
        write_json(root / "adapter_manifest.json", manifest)
        return manifest

    def steps(self, profile, *, train=True, split=None):
        device = profile["parameters"]["device"]
        probe = ("import json,torch,numpy,scipy,PIL,matplotlib; from pathlib import Path; "
                 "import torchvision; " if device == "cuda" else
                 "import json,torch,numpy,scipy,PIL,matplotlib; from pathlib import Path; ")
        probe += ("assert torch.cuda.is_available(), 'CUDA unavailable'; " if device == "cuda" else "")
        probe += ("p={'torch':torch.__version__,'cuda':torch.version.cuda,"
                  "'modules':{m.__name__:str(Path(m.__file__).resolve()) for m in (torch,numpy,scipy,PIL,matplotlib)}}; ")
        if device == "cuda":
            probe += "p['torchvision']=torchvision.__version__; p['modules']['torchvision']=str(Path(torchvision.__file__).resolve()); "
        if profile["adapter_id"] == "neural_ode":
            probe += "import torchdiffeq; p['author_package']=str(Path(torchdiffeq.__file__).resolve()); "
        probe += "Path('import_provenance.json').write_text(json.dumps(p),encoding='utf-8'); print(json.dumps(p))"
        steps = [{"id": "import_check", "argv": ["python", "-c", probe], "timeout_s": 60}]
        if train:
            steps.append({"id": "train", "argv": ["python", "-u", "run_experiment.py"], "timeout_s": 1200})
            steps.append(self.evaluation_step(profile, split))
        for step in steps:
            step["env"] = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "MPLBACKEND": "Agg",
                           "PYTHONHASHSEED": str(profile["parameters"]["seed"]), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
            if device == "cpu":
                step["env"]["CUDA_VISIBLE_DEVICES"] = ""
        return steps

    def evaluation_step(self, profile, split=None):
        split = split or ("fit" if profile["parameters"]["protocol"] == "official_fit" else "validation")
        return {"id": "evaluate", "argv": ["python", "-u", "evaluate.py", split], "timeout_s": 90,
                "env": {"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "2"}}

    def verify_environment(self, profile, execution, workspace):
        root = Path(workspace)
        imported = read_json(root / "import_provenance.json")
        deps = Path(execution["environment"]["dependencies_path"]).resolve()
        if any(not Path(p).resolve().is_relative_to(deps) for p in imported["modules"].values()):
            raise ValueError("依赖从冻结缓存之外导入")
        if profile["parameters"]["device"] == "cuda" and (
                imported["torch"] != "2.5.1+cu121" or imported["cuda"] != "12.1"
                or imported.get("torchvision") != "0.20.1+cu121"):
            raise ValueError("CUDA 运行环境与冻结版本不一致")
        if profile["adapter_id"] == "neural_ode" and (imported["torch"] != "2.5.1+cpu"
                or Path(imported.get("author_package", "")).resolve() != (root / "torchdiffeq/__init__.py").resolve()):
            raise ValueError("Neural ODE 未从固定作者仓库与 CPU 环境导入")

    def verify(self, profile, execution, workspace, snapshot, manifest, spec_hash, split="fit"):
        from src.repository_reproduction import execution_succeeded
        root = Path(workspace)
        if not execution_succeeded(execution):
            raise ValueError("必需执行步骤失败，不能接受指标")
        for name, expected in {**snapshot["files"], **manifest["files"]}.items():
            path = workspace_path(root, name, "protocol file", must_exist=True)
            if digest(path) != expected:
                raise ValueError(f"实验源码/协议被改写: {name}")
        self.verify_environment(profile, execution, workspace)
        training = read_json(root / "artifacts" / "training.json")
        metrics = read_json(root / "artifacts" / f"metrics_{split}.json")
        reported = json.loads(execution["final"]["stdout"].strip().splitlines()[-1])
        if reported != metrics:
            raise ValueError("落盘指标与本次独立评估进程的输出不一致")
        for doc in (training, metrics):
            if doc["spec_sha256"] != spec_hash or doc["prediction_sha256"] != digest(root / "artifacts" / "prediction.npy"):
                raise ValueError("产物不属于本次冻结实验")
        if (training["parameters"] != profile["parameters"]
                or training["steps_completed"] != profile["parameters"]["steps"]
                or len(training["losses"]) != training["steps_completed"]
                or not all(math.isfinite(x) and x >= 0 for x in training["losses"])
                or training["checkpoint_sha256"] != digest(root / "artifacts" / "checkpoint.pt")):
            raise ValueError("训练步数、参数或 checkpoint 核验失败")
        if metrics["split"] != split or not metrics["pass"]:
            raise ValueError("评估范围不符")
        required = set(profile["paper"]["required_metrics"])
        if set(metrics["metrics"]) != required or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in metrics["metrics"].values()):
            raise ValueError("缺少必需指标或指标无效")
        records = [{"name": k, "value": v, "unit": "dB" if k == "psnr" else "scalar",
                    "direction": "maximize" if k == "psnr" else "minimize", "split": split,
                    "stage": "eval", "seed": training["seed"], "source": f"artifacts/metrics_{split}.json",
                    "spec_sha256": spec_hash} for k, v in metrics["metrics"].items()]
        quality = split != "fit" or metrics["metrics"].get("psnr", 100) >= profile["validation"].get("min_psnr_db", 0)
        return {"status": "method_experiment_completed" if quality else "quality_target_not_met",
                "result_level": "method_experiment_completed" if quality else "experiment_completed",
                "is_reproduced": None, "optimization_eligible": True, "quality_pass": quality,
                "reason": profile["validation"]["note"], "scope": profile["validation"]["scope"],
                "protocol_pass": True, "independent_metrics_pass": True,
                "metrics_comparison": {"paper": {}, "actual": metrics["metrics"]}, "metric_records": records,
                "training_summary": {"initial_loss": training["losses"][0], "final_loss": training["losses"][-1],
                    "steps_completed": training["steps_completed"], "training_elapsed_s": training["training_elapsed_s"]}}


class NeuralODEAdapter(SirenAdapter):
    runtime = "neural_ode_runtime.py"
    evaluator = "evaluate_neural_ode.py"

    def prepare_dataset(self, root, profile, workspace, *, offline=False):
        content = json.dumps(profile["dataset"], sort_keys=True, separators=(",", ":")).encode()
        path = Path(workspace) / "dataset_definition.json"
        path.write_bytes(content)
        return {**profile["dataset"], "path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest(), "verified": True}

    def materialize(self, workspace, profile, spec_hash):
        root = Path(workspace)
        raw = (root / "examples/ode_demo.py").read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != profile["source_sha256"]:
            raise ValueError("固定 Neural ODE 源码校验失败")
        source = raw.decode("utf-8")
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "ODEFunc")
        (root / "author_model.py").write_text("import torch\nfrom torch import nn\n\n" + ast.get_source_segment(source,node) + "\n",encoding="utf-8")
        return self._write_runtime(root,profile,spec_hash,["author_model.py","dataset_definition.json"])

    def public_sources(self, workspace, profile):
        source = (Path(workspace) / "examples/ode_demo.py").read_text(encoding="utf-8")
        return [{"source_id": "author_ode_demo", "url": profile["paper"]["reference_source"],
                 "locator": "examples/ode_demo.py", "content": source}]

    def verify(self, *args, **kwargs):
        result = super().verify(*args, **kwargs)
        training = read_json(Path(args[2]) / "artifacts/training.json")
        result["training_summary"].update({k: training[k] for k in ("initial_fit_mae","nfe_training","nfe_evaluation")})
        if kwargs.get("split", "fit") == "fit":
            quality = result["metrics_comparison"]["actual"]["mae"] < training["initial_fit_mae"]
            result["quality_pass"] = quality
            if not quality:
                result.update(status="quality_target_not_met", result_level="experiment_completed")
        return result
