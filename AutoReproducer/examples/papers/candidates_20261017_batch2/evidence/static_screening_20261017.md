# Next independent PDF candidate: static screening only

Reviewed on 2026-10-17. No author training, environment installation, product execution, product edit, registry edit, or Git commit was performed for this research. The previous GCNH -> GREAD -> Critical Look order and failures are unchanged.

## Recommended candidate, pending ordinary registration and freeze gates

**Polynormer: Polynomial-Expressive Graph Transformer in Linear Time**. Chenhui Deng, Zichao Yue, Zhiru Zhang; ICLR 2024. Identity: `arxiv:2403.01232` (downloaded PDF is v3, 6 April 2024). No title, alias, source identity, or hash match was found in the current 55-paper registry; no title/arXiv match was found in the committed docs and examples. This is not a statement that an independent trial has passed.

- PDF: https://arxiv.org/pdf/2403.01232 ; saved as `polynormer.pdf`; 26 pages, SHA-256 `b48d05ad91bc923b4406b3cc4d29295ccc3ea23ffebfb82fc5fcdaa0af48bb7c`.
- Author repo: https://github.com/cornell-zhang/Polynormer ; pinned HEAD `fc8c276c9c5dfbd616d83f65338a3392188a5e08` (2024-04-05). The PDF p.1 explicitly says its source code is freely available there, and its URI annotation has the same complete HTTPS link.
- Classification: Amazon Photo, 7,650 nodes, 745 numerical features, 8 classes. Author split `data/amazon-photo_split.npz` is already in the pinned repo: train 4,590 / validation 1,530 / test 1,530.
- Public raw data: https://raw.githubusercontent.com/shchur/gnn-benchmark/master/data/npz/amazon_electronics_photo.npz ; downloaded successfully, 17,113,960 bytes, SHA-256 `bdb1feb8e6ff42ee44024b04479145029c563fd42fc31bff80af20887ba0439a`. Static NumPy reads confirmed the above shape/classes. These are the public raw arrays, not generated training results.
- CPU support is explicit in author `parse.py:17` (`--cpu`) and `main.py:35-38` (CPU override and automatic CPU fallback). There is no unconditional `torch.cuda.set_device` before the fallback. Torch CUDA seed calls are present; they are not device placement.
- Author full command is the Amazon Photo line in `run.sh:12`, run from repo root. Keeping `$GPU` as 0 on this CPU-only runtime preserves all training arguments:

```text
python main.py --dataset amazon-photo --hidden_channels 64 --local_epochs 200 --global_epochs 1000 --lr 0.001 --runs 1 --local_layers 7 --global_layers 2 --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8 --device 0 --save_model
```

This is the author's **one-run** command, seed 42 by default, with all **1,200 epochs**, 200 local and 1,000 local-to-global, and validation accuracy model selection. Do not increase/decrease repetitions, change seeds, remove ReLU, or shorten epochs to make it pass. `main.py:99` has a fixed full epoch loop, not early stopping. Do not present this as the ten-run paper protocol.

## Paper comparison

PDF p.7, **Table 1** reports test classification accuracy (%) averaged over **10 runs**, with **standard deviation**. For Photo: `Polynormer-r 96.46 +/- 0.26`; plain Polynormer is `96.11 +/- 0.23`. The checked-in source includes ReLU, and README explicitly says its code/expected first-run results use Polynormer-r. Therefore a one-run result can be `reference_only` against the **Polynormer-r** paper row, with differences in repetitions, CPU vs paper GPU hardware, and runtime/dependency versions displayed. It must not be paired with the plain Polynormer row or called independent numerical reproduction.

The author log has several distinct statistics. Use `Final Test` (test accuracy at the best validation accuracy), not `Highest Test`. `logger.py:59` prints mean and `torch.std`; with one run the latter has no finite sample standard deviation and may be `nan`. Preserve mean/run-count semantics, omit or mark SD unavailable if necessary, and do not invent zero SD. A finite traced mean and finite difference are still possible.

## Python and dependency risks

README advertises a Python 3.9 / PyTorch 2.0 / PyG 2.3 Linux CUDA environment, but does not pin an incompatible Python in a lock/requirements file. PyPI currently provides `torch-2.0.1-cp311-cp311-win_amd64.whl`; PyG 2.3.1 declares Python >=3.7. Thus the narrow statically supported Windows choice is **CPython 3.11**, not reproducing the author's Python 3.9 environment and not assuming the old Torch wheel supports 3.12. Current supported newer Torch/PyG combinations could also work but have not been tested by this static research.

`dataset.py` imports `ogb` and `gdown` at module level even for Amazon Photo, despite README calling them dataset-specific. They must be included in real dependency preflight. SciPy, NumPy and scikit-learn are also imported. `torch.load` reloads a tensor/optimizer checkpoint during the local/global transition. No custom CUDA extension is required by the small-data source itself, but real dependency health/native operations remain unverified. CPU performance of seven 8-head GAT layers over 1,200 epochs could be substantial; no runtime estimate was obtained by shortening/trial training.

## Exclusions (not a candidate order)

| Paper/repo | Static reason to exclude from this next supported batch |
| --- | --- |
| FSGNN, sunilkmaurya/FSGNN, `486ff889eb6002b70564c2c57ec3087e41edc745` | Already retired in the registry under its full title **Improving Graph Neural Networks with Simple Architecture Design**, arXiv 2105.07634. `node_class.py:49-50` also hardcodes CUDA. Never register FSGNN as a new alias-independent paper. |
| LINKX / Large Scale Learning on Non-Homophilous Graphs, CUAI/Non-Homophily-Large-Scale, `82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6` | Author CPU branch exists, but official environment explicitly requires Python 3.8, Torch 1.7.1, PyG 1.6.3 and old native extension pins. Not within currently declared Python 3.11/3.12 native support; upgrading this stack before independent acceptance would be a different compatibility effort. |
| CoBFormer / Less is More: on the Over-Globalizing Problem in Graph Transformers, BUPT-GAMMA/CoBFormer, `a2a30ab9608854ac77c06079feb5858682401a7a` | `main.py:53` unconditionally calls `torch.cuda.set_device` before its nominal CPU fallback. Also official Python 3.10 / CUDA environment. No author-supported working CPU entrypoint in this path. |
| TabM, yandex-research/tabm, `28e47ae301c92ec37787dde1ce923a0793f405b4` | Its current package supports Python >=3.9 and CPU, but paper reproduction code explicitly fixes Python 3.10.12 / Torch 2.0.1 and offers CPU setup only on Linux/macOS. Package notebook must not silently replace the paper commands. Keep it out of this Windows 3.11/3.12 paper batch pending a separate compatibility decision. |

Only one next candidate is recommended. No trial order is being registered in this report; registration, current-source freeze, CI and actual browser/API acceptance still belong to the parent task.
