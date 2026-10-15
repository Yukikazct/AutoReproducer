"""Read-only verification of the saved synthetic Caffe GPU check, not mAP."""
import json
import math
from pathlib import Path
import re
import sys

path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parent.parent / 'issue27_rfcn_native_build_20261015_05.json')
record = json.loads(path.read_text('utf-8'))
data = record['data']
assert data['training_executed'] is False
assert data['build_exit_code'] == 0 and data['gpu_probe']['exit_code'] == 0
probe = data['operator_probe']
assert probe['exit_code'] == 0
output = probe['stdout'] + probe['stderr']
assert 'Use GPU with device ID 0' in output
assert 'GPU device name: NVIDIA GeForce RTX 4060 Laptop GPU' in output
values = [float(v) for v in re.findall(r'\bchecksum\s*=\s*([0-9.eE+-]+)', output)]
expected = 2 * 7 * 7 * 0.5
assert len(values) == 2 and all(math.isfinite(v) and abs(v - expected) < 1e-6 for v in values)
cleanup = data['native_cleanup']
assert not cleanup['errors'] and not cleanup['remaining_images'] and not cleanup['remaining_builder_volumes']
assert cleanup['preexisting_images_retained'] is True
print(json.dumps({'synthetic_gpu_operator_verified': True, 'checksums': values,
                  'paper_training_executed': False}, ensure_ascii=False))
