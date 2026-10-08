"""Reviewed experiment adapters; an unknown profile must never use another evaluator."""
from typing import Protocol


class ExperimentAdapter(Protocol):
    def required_files(self, profile): ...
    def prepare_dataset(self, root, profile, workspace, *, offline=False): ...
    def metric_records(self, execution, spec_hash): ...
    def verify_protocol(self, profile, execution, workspace, snapshot, dataset): ...
    def recompute_metrics(self, execution, workspace, dataset, records): ...
    def validate(self, profile, execution, records, error=""): ...


class DLinearAdapter:
    """Keep the existing public helpers and their test/integration injection points."""
    def required_files(self, profile):
        return ["run_longExp.py", *profile["repository_map"].values()]

    def prepare_dataset(self, *args, **kwargs):
        from src.repository_reproduction import prepare_dataset
        return prepare_dataset(*args, **kwargs)

    def metric_records(self, *args):
        from src.repository_reproduction import dlinear_metric_records
        return dlinear_metric_records(*args)

    def verify_protocol(self, *args):
        from src.repository_validation import verify_dlinear_protocol
        return verify_dlinear_protocol(*args)

    def recompute_metrics(self, *args):
        from src.repository_reproduction import recompute_dlinear_metrics
        return recompute_dlinear_metrics(*args)

    def validate(self, *args):
        from src.repository_reproduction import validate_repository
        return validate_repository(*args)

    def public_sources(self, root):
        from src.repository_public_sources import RepositoryPublicSources
        return RepositoryPublicSources(root / "public_source_cache" / "dlinear-2205.13504v3")

    def analysis(self, llm, logger, **kwargs):
        from src.repository_analysis import RepositoryAnalysis
        return RepositoryAnalysis(llm, logger, **kwargs)


_ADAPTERS = {"dlinear": DLinearAdapter}


def get_adapter(profile) -> ExperimentAdapter:
    name = profile.get("adapter_id", "dlinear")
    if name == "siren":
        from src.method_adapters import SirenAdapter
        return SirenAdapter()
    if name not in _ADAPTERS:
        raise ValueError(f"未注册的论文适配器: {name}")
    return _ADAPTERS[name]()
