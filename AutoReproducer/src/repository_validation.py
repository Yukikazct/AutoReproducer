"""Read-only evidence checks for the selected DLinear paper experiment.

This module uses only the standard library. It never imports author code,
unpickles checkpoints, installs dependencies, or executes log contents.
"""
import ast
import hashlib
import math
import re
import struct
from pathlib import Path


_ARGUMENTS = {
    "is_training": 1, "model": "DLinear", "data": "ETTh1", "features": "M",
    "seq_len": 336, "pred_len": 96, "enc_in": 7, "batch_size": 32,
    "learning_rate": 0.005, "train_epochs": 10, "patience": 3, "itr": 1,
    "individual": False, "train_only": False, "use_amp": False,
    "use_gpu": False, "embed": "timeF", "lradj": "type1",
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_inside(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("证据文件路径必须位于本次工作区内")
    target = root / relative
    for parent in (target, *target.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError("证据文件不能经过符号链接")
    resolved = target.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("证据文件缺失或越出工作区")
    return resolved


def _namespace(stdout):
    lines = re.findall(r"^Namespace\(.*\)$", stdout, re.M)
    if len(lines) != 1 or len(lines[0]) > 65536:
        raise ValueError("需要本次唯一的实际Namespace参数日志")
    call = ast.parse(lines[0], mode="eval").body
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
            or call.func.id != "Namespace" or call.args):
        raise ValueError("参数日志必须是Namespace的字面量关键词参数")
    values = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in values:
            raise ValueError("Namespace不允许展开或重复参数")
        values[keyword.arg] = ast.literal_eval(keyword.value)
    return values


def _attribute_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _attribute_name(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def _source_seed(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "fix_seed"
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("固定作者入口缺少唯一fix_seed定义")
    seed = ast.literal_eval(assignments[0].value)
    if type(seed) is not int or seed != 2021:
        raise ValueError("固定作者入口的seed不是2021")
    calls = {name: 0 for name in ("random.seed", "torch.manual_seed", "np.random.seed")}
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        name = _attribute_name(call.func)
        if name in calls and (len(call.args) == 1 and not call.keywords
                              and isinstance(call.args[0], ast.Name)
                              and call.args[0].id == "fix_seed"):
            calls[name] += 1
    if any(count != 1 for count in calls.values()):
        raise ValueError("作者入口没有同时固定Python、NumPy和PyTorch随机种子")
    return seed


def _npy_info(path):
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("预测文件不是NPY格式")
        version = stream.read(2)
        if version == b"\x01\x00":
            length_size, length_format, encoding = 2, "<H", "latin1"
        elif version in (b"\x02\x00", b"\x03\x00"):
            length_size, length_format = 4, "<I"
            encoding = "utf-8" if version[0] == 3 else "latin1"
        else:
            raise ValueError("不支持的NPY格式版本")
        raw_length = stream.read(length_size)
        if len(raw_length) != length_size:
            raise ValueError("NPY头部被截断")
        length = struct.unpack(length_format, raw_length)[0]
        if not 0 < length <= 65536:
            raise ValueError("NPY头部长度不合理")
        raw_header = stream.read(length)
        if len(raw_header) != length:
            raise ValueError("NPY头部被截断")
        header = ast.literal_eval(raw_header.decode(encoding).strip())
        if (not isinstance(header, dict)
                or set(header) != {"descr", "fortran_order", "shape"}):
            raise ValueError("NPY头部字段不符合数组协议")
        shape = header["shape"]
        if (not isinstance(shape, tuple)
                or any(type(value) is not int or value <= 0 for value in shape)):
            raise ValueError("预测数组shape必须是正整数元组")
        if shape != (2785, 96, 7):
            raise ValueError(f"预测数组shape错误: {shape}，需要(2785, 96, 7)")
        dtype = header["descr"]
        if not isinstance(dtype, str) or not re.fullmatch(r"[<>=|]f[48]", dtype):
            raise ValueError("预测数组必须使用无pickle的浮点dtype")
        if header["fortran_order"] is not False:
            raise ValueError("预测数组布局不符合作者concat/save输出")
        expected_size = stream.tell() + math.prod(shape) * int(dtype[-1])
    if path.stat().st_size != expected_size:
        raise ValueError("预测数组数据区被截断或文件大小与头部不一致")
    return {"shape": list(shape), "dtype": dtype, "bytes": expected_size}


def verify_dlinear_protocol(profile, execution, workspace, repository, dataset):
    """Verify a full author-script experiment independently of metric agreement.

    The original 10-epoch maximum and patience-3 early stopping both count as
    completion. Returned checks and artifact hashes can be persisted as evidence.
    """
    checks, epochs_completed, artifacts = [], 0, {}
    root = Path(workspace)
    stdout = (execution.get("final") or {}).get("stdout") or ""

    def check(name, operation):
        try:
            details = operation()
            checks.append({"name": name, "pass": True, "detail": details})
        except (OSError, ValueError, TypeError, KeyError, SyntaxError, UnicodeError,
                struct.error, RecursionError) as exc:
            checks.append({"name": name, "pass": False, "detail": str(exc)})

    def execution_check():
        final = execution.get("final") or {}
        steps = execution.get("steps") or execution.get("stages") or execution.get("attempts") or []
        if (execution.get("success") is not True or final.get("success") is not True
                or final.get("exit_code") != 0 or final.get("timed_out") or final.get("cancelled")):
            raise ValueError("本次训练/测试没有成功退出")
        required_ids = [step["id"] for step in profile["steps"] if step.get("required") is not False]
        required = [step for step in steps if step.get("required") is not False]
        if [step.get("id") for step in required] != required_ids:
            raise ValueError("实际必需步骤缺失、重复或顺序与冻结计划不符")
        if any(step.get("success") is not True or step.get("exit_code") != 0
               or step.get("timed_out") or step.get("cancelled") for step in required):
            raise ValueError("至少一个必需步骤失败、超时或取消")
        return {"required_steps": required_ids, "exit_code": 0}

    def source_check():
        nonlocal root
        if root.is_symlink() or not root.is_dir():
            raise ValueError("独立源码工作区缺失或使用符号链接")
        root = root.resolve(strict=True)
        requested = profile["repository"]
        if (repository.get("resolved_sha") != requested["revision"]
                or repository.get("revision") != requested["revision"]
                or repository.get("url", "").rstrip("/").removesuffix(".git")
                != requested["url"].rstrip("/").removesuffix(".git")):
            raise ValueError("仓库来源或固定commit与冻结预设不一致")
        if Path(repository["path"]).resolve() != root:
            raise ValueError("仓库记录指向其他工作区")
        files = repository.get("files") or {}
        if not isinstance(files, dict) or not files:
            raise ValueError("缺少固定commit文件SHA-256清单")
        required = {"run_longExp.py", *profile["repository_map"].values()}
        if not required.issubset(files):
            raise ValueError("源码清单缺少作者入口、方法或评估文件")
        for relative, expected in files.items():
            if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
                raise ValueError(f"无效源码校验和: {relative}")
            if _sha256(_file_inside(root, relative)) != expected:
                raise ValueError(f"训练后的源码与固定commit不一致: {relative}")
        return {"resolved_sha": requested["revision"], "verified_files": len(files)}

    def arguments_check():
        values = _namespace(stdout)
        for name, expected in _ARGUMENTS.items():
            actual = values.get(name)
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(f"实际参数{name}={actual!r}，需要{expected!r}")
        parameters = profile["parameters"]
        for name in ("seq_len", "pred_len", "train_epochs", "patience", "itr", "seed"):
            expected = 2021 if name == "seed" else _ARGUMENTS[name]
            if type(parameters.get(name)) is not type(expected) or parameters[name] != expected:
                raise ValueError(f"预设{name}不是选定论文完整实验参数")
        seed = _source_seed(_file_inside(root, "run_longExp.py"))
        target = root / profile["dataset"]["target"]
        actual_data = root / values["root_path"] / values["data_path"]
        if actual_data.resolve() != target.resolve():
            raise ValueError("实际入口使用的数据路径与冻结数据不一致")
        return {"parameters": {name: values[name] for name in _ARGUMENTS}, "source_seed": seed}

    def trace_check():
        nonlocal epochs_completed
        training_marker, test_marker = ">>>>>>>start training :", ">>>>>>>testing :"
        if stdout.count(training_marker) != 1 or stdout.count(test_marker) != 1:
            raise ValueError("需要本次唯一官方训练与最终test阶段")
        start, finish = stdout.index(training_marker), stdout.index(test_marker)
        if start >= finish:
            raise ValueError("训练与测试阶段顺序错误")
        training, test = stdout[start:finish], stdout[finish:]
        for label, expected in (("train", 8209), ("val", 2785), ("test", 2785)):
            counts = re.findall(rf"^{label} (\d+)\s*$", training, re.M)
            if counts != [str(expected)]:
                raise ValueError(f"官方{label}窗口数不符: {counts}")
        if re.findall(r"^test (\d+)\s*$", test, re.M) != ["2785"]:
            raise ValueError("最终test未使用全部2785个官方窗口")
        summaries = list(re.finditer(r"^Epoch: (\d+), Steps: (\d+) \| Train Loss:", training, re.M))
        numbers = [int(match.group(1)) for match in summaries]
        epochs_completed = len(numbers)
        if not numbers or numbers != list(range(1, epochs_completed + 1)):
            raise ValueError("训练epoch日志缺失、重复或不连续")
        if any(match.group(2) != "256" for match in summaries):
            raise ValueError("每轮训练未完成官方256个batch")
        if not 1 <= epochs_completed <= 10:
            raise ValueError("训练轮数不符合官方10轮上限")
        stops = list(re.finditer(r"^Early stopping\s*$", training, re.M))
        if len(stops) > 1 or (stops and stops[0].start() < summaries[-1].end()):
            raise ValueError("官方early stopping位置或次数异常")
        if epochs_completed < 10:
            tail = training[summaries[-1].end():]
            if (epochs_completed < 3 or len(stops) != 1
                    or not re.search(r"^EarlyStopping counter: 3 out of 3\s*\nEarly stopping\s*$", tail, re.M)):
                raise ValueError("训练提前结束但没有官方patience=3早停证据")
        return {"epochs_completed": epochs_completed, "early_stopping": bool(stops),
                "train_windows": 8209, "validation_windows": 2785, "test_windows": 2785}

    def dataset_check():
        expected = profile["dataset"]
        path = _file_inside(root, expected["target"])
        actual_hash = _sha256(path)
        if (dataset.get("sha256") != expected["sha256"] or actual_hash != expected["sha256"]
                or path.stat().st_size != expected["bytes"]
                or Path(dataset["path"]).resolve() != path or dataset.get("kind") != "real"):
            raise ValueError("实际训练数据的来源路径、大小或SHA-256不符")
        return {"path": str(path), "sha256": actual_hash, "bytes": path.stat().st_size}

    def artifacts_check():
        for name in ("checkpoint.pth", "pred.npy"):
            paths = list(root.rglob(name))
            if len(paths) != 1:
                raise ValueError(f"需要本次唯一{name}产物，实际{len(paths)}个")
            path = _file_inside(root, paths[0].relative_to(root))
            if not path.stat().st_size:
                raise ValueError(f"{name}产物为空")
            info = {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            if name == "pred.npy":
                info.update(_npy_info(path))
            artifacts[name] = info
        return artifacts

    for name, operation in (("required_steps", execution_check), ("fixed_source", source_check),
                            ("actual_parameters_and_seed", arguments_check), ("full_training_trace", trace_check),
                            ("real_dataset", dataset_check), ("checkpoint_and_predictions", artifacts_check)):
        check(name, operation)
    passed = all(item["pass"] for item in checks)
    failures = [f"{item['name']}: {item['detail']}" for item in checks if not item["pass"]]
    return {"pass": passed, "checks": checks, "scope": "selected_paper_experiment",
            "epochs_completed": epochs_completed, "artifacts": artifacts,
            "reason": "固定作者源码、真实数据和完整训练/test协议已核验" if passed else "; ".join(failures)}
