"""Count native optimizer updates in the author notebook kernel."""
import atexit
import json
import os
from pathlib import Path

_AUTOREPRO_STEPS = 0


def _autorepro_after_step(optimizer, args, kwargs):
    global _AUTOREPRO_STEPS
    _AUTOREPRO_STEPS += 1


def _autorepro_write_steps():
    try:
        identifier = os.environ['AUTOREPRO_NOTEBOOK_STEP']
        path = Path(os.environ['AUTOREPRO_CAPTURE_DIR']) / ('notebook_steps_' + identifier + '.json')
        path.write_text(json.dumps({'optimizer_steps': _AUTOREPRO_STEPS}), encoding='utf-8')
    except (OSError, ValueError, KeyError):
        pass


try:
    from torch.optim.optimizer import register_optimizer_step_post_hook
    _AUTOREPRO_HANDLE = register_optimizer_step_post_hook(_autorepro_after_step)
    atexit.register(_autorepro_write_steps)
except (ImportError, AttributeError):
    pass
