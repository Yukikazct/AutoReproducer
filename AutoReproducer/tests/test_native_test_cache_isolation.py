"""Optional native tests cannot silently borrow a live disposable app cache."""
from pathlib import Path

from discovered_fixtures import configured_native_cache
from src.runtime_platform import runtime_fingerprint


def test_no_configuration_does_not_scan_application_cache(monkeypatch):
    monkeypatch.delenv('AUTOREPRO_TEST_CPU_DEPS', raising=False)
    def unexpected_scan(*args, **kwargs):
        raise AssertionError('native tests must not scan real application caches')
    monkeypatch.setattr(Path, 'glob', unexpected_scan)
    assert configured_native_cache('AUTOREPRO_TEST_CPU_DEPS', '2.5.1+cpu') is None


def test_production_cache_is_rejected_before_inspecting_its_packages(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'data/deps/repository' / runtime_fingerprint() / 'live'
    monkeypatch.setenv('AUTOREPRO_TEST_CPU_DEPS', str(path))
    def forbidden(*args, **kwargs):
        raise AssertionError('production packages must not be inspected')
    monkeypatch.setattr(Path, 'is_file', forbidden)
    assert configured_native_cache('AUTOREPRO_TEST_CPU_DEPS', '2.5.1+cpu') is None


def test_only_explicit_ready_matching_test_cache_is_selected(tmp_path, monkeypatch):
    path = tmp_path / 'deps/repository' / runtime_fingerprint() / 'test-only'
    path.mkdir(parents=True)
    monkeypatch.setenv('AUTOREPRO_TEST_CPU_DEPS', str(path))
    assert configured_native_cache('AUTOREPRO_TEST_CPU_DEPS', '2.5.1+cpu') is None
    (path / '.ready').touch()
    (path / 'torch-2.5.1+cpu.dist-info').mkdir()
    (path / 'numpy-1.26.4.dist-info').mkdir()
    assert configured_native_cache('AUTOREPRO_TEST_CPU_DEPS', '2.5.1+cpu') == path
    assert configured_native_cache('AUTOREPRO_TEST_CPU_DEPS', '2.5.1+cu121') is None
