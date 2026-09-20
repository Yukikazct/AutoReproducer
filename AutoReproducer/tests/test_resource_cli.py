"""ResourceCLI 三层缓存命令化测试（P2）。

通过 subprocess 以真实进程执行 scripts/resource_cli.py，验证：
1. status / list / manifest 只读视图；
2. archive -> restore L1 温存储往返；
3. prune 默认 dry_run 只建议不删；--yes 先归档 L1 再清理 L0；
4. quota-check 剩余配额足够放行（0）；不足拒绝（2）并给建议。

独立 AUTOREPRO_DATA_ROOT 隔离，不触碰真实 data/。

运行: python -m pytest tests/test_resource_cli.py -v
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _PROJECT_ROOT / "scripts" / "resource_cli.py"
_PY = sys.executable


@pytest.fixture()
def cli_env(tmp_path):
    """独立数据根 + 小配额（1MB）便于触发超限，返回 (env, data_root)。"""
    data_root = tmp_path / "data"
    env = dict(os.environ)
    env["AUTOREPRO_DATA_ROOT"] = str(data_root)
    env["AUTOREPRO_L0_QUOTA_GB"] = "0.001"      # 1MB
    return env, data_root


def run_cli(env, *argv):
    return subprocess.run(
        [_PY, str(_SCRIPT), *argv], capture_output=True,
        text=True, env=env, timeout=120)


def _seed_paper(data_root: Path, pid: str, title: str = "Test Paper") -> None:
    """直接播种一篇论文的 L0 资源与 manifest（绕过网络）。"""
    from src.resource_manager import ResourceManager
    mgr = ResourceManager(data_root=str(data_root), quota_bytes=1024 * 1024)
    repo = mgr._repo_dir(pid)
    (repo / "train.py").parent.mkdir(parents=True, exist_ok=True)
    (repo / "train.py").write_text("# dummy\n", encoding="utf-8")
    table = mgr._dataset_dir(pid) / "dataset_smoke"
    table.mkdir(parents=True, exist_ok=True)
    (table / "samples.csv").write_text(
        "id,feature\n0,0.1\n", encoding="utf-8")
    mgr.save_manifest(mgr.build_manifest(
        paper_id=pid, paper_title=title,
        code_url="https://github.com/example/repo",
        dataset_name="MNIST"))


# ---------------- 1. 只读视图 ----------------

def test_status_view(cli_env):
    env, data_root = cli_env
    _seed_paper(data_root, "cli000000001")
    res = run_cli(env, "status")
    assert res.returncode == 0
    assert "L0 热缓存占用" in res.stdout
    assert "cli000000001" in res.stdout


def test_list_view(cli_env):
    env, data_root = cli_env
    _seed_paper(data_root, "cli000000002", title="View Paper")
    res = run_cli(env, "list")
    assert res.returncode == 0
    assert "cli000000002" in res.stdout
    assert "View Paper" in res.stdout
    assert "repos" in res.stdout
    assert "dataset_smoke" in res.stdout


def test_manifest_view_missing(cli_env):
    env, _ = cli_env
    res = run_cli(env, "manifest", "no-such-paper")
    assert res.returncode == 1
    assert "不存在" in res.stderr


def test_manifest_view(cli_env):
    env, data_root = cli_env
    _seed_paper(data_root, "cli000000003")
    res = run_cli(env, "manifest", "cli000000003")
    assert res.returncode == 0
    assert '"paper_id": "cli000000003"' in res.stdout
    assert "MNIST" in res.stdout


# ---------------- 2. archive / restore 往返 ----------------

def test_archive_restore_roundtrip(cli_env, tmp_path):
    env, data_root = cli_env
    _seed_paper(data_root, "cli000000004")
    dest = tmp_path / "l1"
    res = run_cli(env, "archive", "cli000000004", "--dest", str(dest))
    assert res.returncode == 0
    assert str(dest / "cli000000004.zip") in res.stdout
    assert (dest / "cli000000004.zip").is_file()

    # 清空 L0 再恢复
    from src.resource_manager import ResourceManager
    mgr = ResourceManager(data_root=str(data_root))
    mgr.cleanup("cli000000004", keep_manifest=True)
    assert not (mgr._repo_dir("cli000000004")).exists()

    res2 = run_cli(env, "restore", str(dest / "cli000000004.zip"))
    assert res2.returncode == 0
    assert "已恢复" in res2.stdout
    assert (mgr._repo_dir("cli000000004") / "train.py").is_file()
    assert (mgr._dataset_dir("cli000000004") / "dataset_smoke"
            / "samples.csv").is_file()


def test_restore_missing_archive(cli_env):
    env, _ = cli_env
    res = run_cli(env, "restore", "/no/such/archive.zip")
    assert res.returncode == 1
    assert "归档不存在" in res.stderr


# ---------------- 3. prune 配额守护 ----------------

def test_prune_dry_run_only_suggests(cli_env):
    env, data_root = cli_env
    _seed_paper(data_root, "cli000000005")
    # 手动放大 L0（配额 1MB）：塞 1.5MB 文件触发超限
    from src.resource_manager import ResourceManager
    mgr = ResourceManager(data_root=str(data_root))
    big = mgr._dataset_dir("cli000000005") / "big.bin"
    big.write_bytes(b"\0" * (1536 * 1024))

    res = run_cli(env, "prune")
    assert res.returncode == 2                      # 超限 -> 非零
    assert "建议" in res.stdout or "dry_run" in res.stdout
    # dry_run：资源仍在 L0
    assert mgr._repo_dir("cli000000005").exists()
    assert res2_exists(mgr, "cli000000005")


def res2_exists(mgr, pid):
    return (mgr._dataset_dir(pid) / "big.bin").exists()


def test_prune_yes_archives_then_cleans(cli_env):
    env, data_root = cli_env
    _seed_paper(data_root, "cli000000006")
    from src.resource_manager import ResourceManager
    mgr = ResourceManager(data_root=str(data_root))
    big = mgr._dataset_dir("cli000000006") / "big.bin"
    big.write_bytes(b"\0" * (1536 * 1024))

    res = run_cli(env, "prune", "--yes")
    assert res.returncode == 0
    assert "已归档并清理" in res.stdout
    # L0 已清，但 L1 归档与 manifest 保留
    assert not mgr._repo_dir("cli000000006").exists()
    assert not (mgr._dataset_dir("cli000000006") / "big.bin").exists()
    assert (mgr.archive_root / "cli000000006.zip").is_file()
    manifest = mgr.get_manifest("cli000000006")
    assert manifest and manifest.get("cleaned_at")


def test_prune_within_quota_noop(cli_env):
    env, data_root = cli_env
    _seed_paper(data_root, "cli000000007")
    env2 = dict(env)
    env2["AUTOREPRO_L0_QUOTA_GB"] = "5"             # 放宽配额
    res = run_cli(env2, "prune")
    assert res.returncode == 0
    assert "无需清理" in res.stdout


# ---------------- 4. quota-check 拒绝下载 ----------------

def test_quota_check_ok_when_enough(cli_env):
    env, _ = cli_env
    res = run_cli(env, "quota-check", "--size-gb", "0.0005")
    assert res.returncode == 0
    assert "OK" in res.stdout


def test_quota_check_refuses_when_exceeded(cli_env):
    env, data_root = cli_env
    _seed_paper(data_root, "cli000000008")
    from src.resource_manager import ResourceManager
    mgr = ResourceManager(data_root=str(data_root))
    big = mgr._dataset_dir("cli000000008") / "big.bin"
    big.write_bytes(b"\0" * (900 * 1024))           # 已用 ~0.9MB/1MB

    res = run_cli(env, "quota-check", "--size-gb", "0.5")
    assert res.returncode == 2                      # 拒绝下载
    assert "配额不足" in res.stderr
    assert "建议先归档" in res.stderr
    assert "cli000000008" in res.stderr


def test_subcommand_required(cli_env):
    env, _ = cli_env
    res = run_cli(env)
    assert res.returncode != 0