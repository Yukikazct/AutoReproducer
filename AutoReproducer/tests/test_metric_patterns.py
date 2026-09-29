"""指标提取正则测试（iTransformer 风格输出：mse/mae/smape）。

背景缺陷：iTransformer 官方代码输出 `mse:0.428, mae:0.421`，而
_METRIC_PATTERNS 只有 accuracy/f1/precision/recall/loss/rmse/mse，
mae 永远提取不到 -> VALIDATE 判"声明了但未提取到"。

运行: python -m pytest tests/test_metric_patterns.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.result_validator import (  # noqa: E402
    ResultValidatorAgent,
)
from src.metric_keys import norm_metric_key  # noqa: E402


def _extract(text: str) -> dict:
    return ResultValidatorAgent._extract_metrics(None, text)


def test_itransformer_style_mse_mae():
    metrics = _extract("mse:0.428, mae:0.421")
    assert metrics["mse"] == 0.428
    assert metrics["mae"] == 0.421


def test_uppercase_space_separated():
    metrics = _extract("MSE: 0.428  MAE: 0.421")
    assert metrics["mse"] == 0.428
    assert metrics["mae"] == 0.421


def test_smape_extracted():
    metrics = _extract("test mse:0.3, mae:0.2, smape:12.5")
    assert metrics["mse"] == 0.3
    assert metrics["mae"] == 0.2
    assert metrics["smape"] == 12.5


def test_rmse_still_not_confused_with_mse():
    metrics = _extract("rmse: 1.2")
    assert metrics.get("rmse") == 1.2
    assert "mse" not in metrics


def test_mean_absolute_error_spelled_out():
    metrics = _extract("Mean Absolute Error: 0.42")
    assert metrics["mae"] == 0.42


def test_norm_metric_key_keeps_mae():
    assert norm_metric_key("MAE") == "mae"
    assert norm_metric_key("mean_absolute_error") == "mean_absolute_error" \
        or norm_metric_key("mean_absolute_error") in (
            "mae", "mean_absolute_error")


def test_equality_style_line():
    metrics = _extract("mse=0.428")
    assert metrics["mse"] == 0.428
