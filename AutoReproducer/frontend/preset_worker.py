"""Continue one preset in the automatically prepared, owned Python runtime."""
import json
import os
import sys

from src.process_lifecycle import termination_signals, watch_parent_session, watch_cancellation_file


@watch_parent_session()
@termination_signals()
def main():
    # Keep credentials out of argv, environment manifests and temporary files.
    payload = sys.stdin.buffer.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError("Preset request exceeds the local pipe limit")
    request = json.loads(payload.decode("utf-8"))
    if not isinstance(request, dict) or request.get("_managed_runtime") is not True:
        raise ValueError("A prepared preset worker requires a managed request")
    if os.name == "posix":
        # Nested runners keep private process groups inside this owned session.
        # The outer owner can still find them after their immediate parent exits.
        os.environ["AUTOREPRO_WORKER_SESSION"] = str(os.getsid(0))
    from frontend.backend_pipeline import run_pipeline_core
    cancel_file, cancel_token = request.pop("_cancel_file", None), request.pop("_cancel_token", None)
    with watch_cancellation_file(cancel_file, cancel_token):
        result = run_pipeline_core(**request)
    return 0 if result.get("state") == "COMPLETED" and not result.get("error") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
