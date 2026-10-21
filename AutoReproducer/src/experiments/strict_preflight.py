"""Probe the frozen device with real optimizer and NumPy operations."""
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch


root = Path(__file__).resolve().parent
contract = json.loads((root / 'config.json').read_text(encoding='utf-8'))['capture']
device = contract.get('device', 'cpu')
model = torch.nn.Linear(2, 2).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
before = model.weight.detach().clone()
values = torch.tensor([[1., 2.], [-1., -2.]], device=device)
labels = torch.tensor([0, 1], device=device)
loss = torch.nn.functional.cross_entropy(model(values), labels)
loss.backward()
optimizer.step()
if not torch.isfinite(loss) or torch.equal(before, model.weight):
    raise ValueError('Native optimizer probe did not update finite parameters')
array = values.detach().cpu().numpy().copy()
if not np.array_equal(torch.from_numpy(array).numpy(), array):
    raise ValueError('Native Torch/NumPy interoperability probe failed')
record = {'python': sys.version, 'platform': platform.platform(), 'device': device,
          'native_probe_pass': True, 'status': 'prepared',
          'checks': ['optimizer_update', 'finite_loss', 'numpy_interoperability'],
          'versions': {'torch': torch.__version__, 'numpy': np.__version__}}
(root / 'preflight.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record), flush=True)
