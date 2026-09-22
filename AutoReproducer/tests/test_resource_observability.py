import json
from src.resource_manager import ResourceManager


def test_dataset_download_emits_event_and_inventory(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOREPRO_DATA_ROOT", str(tmp_path / "data"))
    manager = ResourceManager(data_root=str(tmp_path / "data"),
                             dataset_registry=False)

    result = manager.fetch_dataset("paper001", "SyntheticSet", level="smoke")

    assert result["state"] == "smoke-synth"
    event_file = tmp_path / "data" / "resource_events.jsonl"
    events = [json.loads(line) for line in event_file.read_text().splitlines()]
    assert [event["state"] for event in events] == ["running", "succeeded"]
    assert events[-1]["resource_type"] == "dataset"


def test_cached_dataset_emits_cached_event(tmp_path):
    manager = ResourceManager(data_root=str(tmp_path / "data"),
                             dataset_registry=False)
    manager.fetch_dataset("paper002", "SyntheticSet", level="smoke")
    manager.fetch_dataset("paper002", "SyntheticSet", level="smoke")

    events = manager.resource_events.list()
    assert events[-1]["state"] == "cached"
