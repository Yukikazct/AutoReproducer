# Two additional independent PDF candidates: static screening only

Reviewed 2026-10-17. This report supplements `static_screening_20261017.md` (Polynormer). It does **not** register a candidate batch or freeze a trial order. No author training, package installation, product run, product-source edit, acceptance-registry edit, commit, or push was performed. All newly saved material is under `.tmp/next-pdf-candidates/`. Parent owns the actual registry, freeze, browser, and API acceptance gates.

Static identity check against the current **55-paper** registry found no normalized title/alias/source-ID matches or exact PDF-hash matches for these two papers or Polynormer. This includes `BN-Pool`, `BNPool`, `Bayesian Nonparametric`, `PyGSD`, `PyTorch Geometric Signed [and] Directed`, their arXiv IDs, and MagNet. Registry SHA-256 at check time: `e1c33afcac856f2531ab3a121ef06ec796f588b84893219bb3e9fcd52fd650e4`. Evidence: `static_candidate_identity_check.json`. Recheck before registration; absence from this snapshot does not independently establish the trial's eligibility.

## BN-Pool: strong modern Windows CPU packaging; important data/protocol differences

**Identity:** *BN-Pool: Bayesian Nonparametric Pooling for Graphs*, Daniele Castellana and Filippo Maria Bianchi, TMLR 2026, `arxiv:2501.09821` (downloaded v2, 1 April 2026). Repository title aliases *BNPool: Bayesian Nonparametric Pooling for Graph Neural Networks* and *Bayesian Nonparametric Graph Pooling* refer to this same paper, not separate samples.

- PDF: https://arxiv.org/pdf/2501.09821 ; local `bnpool.pdf`, **32 pages**, SHA-256 `35c416b0b1927db60cd936cd430da1dded1e9794799ed13adfecce0f4c3fa316`. PDF p.1 explicitly identifies the full author repository URL.
- Author repository: https://github.com/NGMLGroup/Bayesian-Nonparametric-Graph-Pooling ; pinned `206bcb4c642468ba4c51f79812ff1e220756937f` (2026-04-01), local `BNPool/`.
- Full authored default command, root cwd, explicitly documented in README lines 80–83:

```text
python run_classification.py
```

Keep this command/configuration unchanged. It is graph classification with **BN-Pool**, default **MUTAG**, fold **0** of a ten-fold partition, seed **777**, batch size **32**. It is **one fold**, not ten independent runs. `config/run_classification.yaml` keeps max epochs **100000**, early stopping active with patience **500**, monitoring **val_loss**, min mode. Adam LR **0.001**, plateau scheduler patience **150**, prior coefficient cosine schedule **300 epochs**. Default pooler k50, alpha_DP10, K_var1, K_mu10, K_init1. No training was shortened to obtain a runtime estimate.

Source `run_classification.py:139-150` explicitly chooses CPU and `devices=1` when CUDA is absent. `trainer.fit` runs naturally, and `trainer.test(..., ckpt_path='best')` uses the validation-selected checkpoint. The classification module logs test accuracy as `test_acc` (`source/pl_modules/classification_module.py:217-227`). Preserve that meaning and the one-fold run count.

### Public input and comparison

**The current default data is not the same size as the paper's table.** `source/data/data_modules.py:147-153` uses `TUDataset(..., cleaned=args.get('clean', True))`; MUTAG's configuration does not override `clean`. Therefore it loads cleaned MUTAG, not the 188-graph uncleaned set.

Downloaded the actual public cleaned archive from https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/MUTAG.zip as `MUTAG_cleaned.zip`: **17,653 bytes**, SHA-256 `ea3576783c61cd2b3a942e9e3bab821cd4b359f5409fd53b31192c721207f6fa`. Static ZIP text inspection found **135 graphs**, binary graph labels -1/+1. The source uses fold0 and then ratio0.85 train/validation (`source/utils/data.py:15-25`), producing **102 train / 19 validation / 14 test** from these 135 graphs. These are static shape/split-count facts, not actual loading/training evidence.

PDF **p.18 Table 3** reports **BN-Pool MUTAG accuracy 88 +/- 7 percent**, mean and **standard deviation**. PDF p.26 Table 10 describes **188 MUTAG graphs**. PDF pp.15–16 uses ten-fold evaluation, LR5e-4, validation accuracy monitoring, scheduler patience30, early-stop patience100, and prior schedule50; Table11 p.27 uses optimal alpha_DP1 for MUTAG, whereas current default is10. Do not silently override these differences or claim the current default one fold is the paper's ten-fold protocol.

An actual traced `test_acc` may be compared **reference_only** to the same task/model/dataset-family paper row **only with the cleaned135 vs raw188 distinction and every other material setting difference displayed**. This data-version mismatch is a real suitability risk: if the product correctly declines even a reference comparison for it, this candidate fails the formal comparison gate; the raw data or plan must not be rewritten merely to pass. No valid numerical reproduction claim follows from executing this command.

### Dependencies

`pyproject.toml:6` explicitly supports **Python >=3.11,<3.13**. The pinned author's package sources explicitly provide **Windows CPU** Torch2.4.1 from `download.pytorch.org/whl/cpu`, and torch-cluster1.6.3, torch-scatter2.1.2, torch-sparse0.6.18 from `data.pyg.org/whl/torch-2.4.0+cpu.html`. Both cp311 and cp312 Windows wheels for all three extensions were found in the public index (saved `torch24_cpu_wheels.html`). PyG2.6.1, torch-geometric-pool1.0.0, lightning2.6.1, numpy1.26.4 and scikit-learn1.8.0 are declared. PyPI metadata for pool1.0.0 permits Torch>=1.8,<3/PyG>=2.6,<3; lightning2.6.1 permits Torch>=2.1,<4. No static version contradiction was found for Windows CPU.

This is wheel availability and authored intent, **not** verified installation, module imports, native kernels, Windows workers, Lightning compatibility, runtime duration, or product planner success. A real run must install the pinned checkout/dependencies and perform native preflight within the one-environment lifecycle. Linux source indexes use CUDA packages despite the runtime CPU fallback; do not claim that index is an author CPU-only Linux lock.

## PyGSD: very small CPU node classification, with author example vs benchmark differences

**Identity:** *PyTorch Geometric Signed Directed: A Software Package on Graph Neural Networks for Signed and Directed Graphs*, Yixuan He, Xitong Zhang, Junjie Huang, Benedek Rozemberczki, Mihai Cucuringu, Gesine Reinert; LoG/PMLR 2024, `arxiv:2202.10793`. Aliases: **PyGSD**, **PyTorch Geometric Signed Directed**, **pytorch_geometric_signed_directed**. This is the software benchmark paper, not the original MagNet or already retired DiGCN paper. The acceptance PDF and table must remain this paper's own identity throughout.

- PDF: https://arxiv.org/pdf/2202.10793 ; local `pygsigneddirected.pdf`, **27 pages**, SHA-256 `3902ece975f46934320629ecdcf5b85c5bdd32a9f3bd9b864c3737bcba997fbc`. PDF p.1 explicitly supplies the matching full author repository URL.
- Author repository: https://github.com/SherylHYX/pytorch_geometric_signed_directed ; pinned `2968b55b3378a01ade9f391e194f0077e2c6ee3b` (2026-09-03), local `PyGSignedDirected/`.
- Full standalone author classification script, root cwd after installing the **pinned checkout** as a package:

```text
python examples/magnet_node.py
```

The script itself is authored and complete, but **README does not spell out this exact command**; README lists the implemented node-classification model and package installation. Whether discovery/planning correctly chooses this file is an untested product gate. Do not manually inject it into a generated plan to rescue acceptance.

`examples/magnet_node.py` defaults to **webkb/cornell**, **200 epochs** for **each of the 10 public mask columns**, LR0.01, weight_decay0.0005, K1, q0.25, fixed q, two layers, hidden width16. Line50 explicitly selects `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. There is no early stopping or validation checkpoint selection in this script: it tests the **last epoch** for each split and resets model parameters afterward. No random seed is set; do not add one. Preserve all **10 x 200 epochs**, and every `Split: NN, Test_Acc: ...` line. There is no final author mean/SD line. If the product derives an aggregate, it must retain all ten original split values, units, run count and stated SD convention; do not use the best split or pretend an aggregate was printed by the author.

The MagNet path uses PyTorch/PyG operations and SciPy utilities; no hardcoded `.cuda()`/CUDA-only extension was found in this entrypoint, `MagNet_node_classification`, `MagNetConv`, or `get_magnetic_Laplacian`. Static AST inspection of the package found external import roots only Torch, PyG, NumPy, SciPy, scikit-learn and NetworkX, all declared by setup.py. This is not a successful import or native-operation test.

### Public input and comparison

The author loader delegates WebKB to PyG (`data/directed/load_directed_real_data.py:3,48-50`) and preserves its masks. Retrieved official PyG2.6.1 WebKB source as `pyg261_webkb.py`; it fetches Geom-GCN public node/edge text and ten NPZ split files. All **12 public files** were downloaded without running author code, to `cornell_raw_static/`. `cornell_static_evidence.json` contains every source URL, byte size, SHA-256, and static split inspection.

Static results: **183 nodes, 1,703 features, 5 classes, 298 raw edge rows**; all ten splits have **87 train / 59 validation / 37 test**, mutually disjoint and covering all183 nodes. Node text SHA-256 `38561ad44e29199a826ccef7f94f469b94a912076f83fbc453a69b8ac2b4b858`; edge text SHA-256 `12dc60defc4f8a1f18499e3695bfd62ef0421f9b722297276e7f02dec7ae3dca`. `static_cornell_probe.py` documents that only standard urllib downloads and `np.load(..., allow_pickle=False)` reads were performed; no author module was imported or executed.

PDF **p.9 Table 5**, **Node classification / MagNet / Cornell = 70.3 +/- 5.1 percent**. Section8 p.8 defines reported uncertainty as **one standard deviation**. Do not substitute Table5's MagNet Cornell direction-prediction79.8 or link-prediction66.9/56.0 results.

Use **reference_only** and explicitly state that this is the author's current example configuration, not the paper's tuned benchmark protocol. Appendix E pp.25–26 says classification adopts original MagNet settings and hyperparameter search, while its runtime measurement uses **10 folds x 500 epochs, 32 filters** and GPU hardware. The current example uses **200 epochs / 16 filters**, fixed q0.25, last-epoch test, present-day pinned source/dependencies, and CPU. The Appendix's 3000 epochs/500 patience statement refers specifically to **link prediction**, so do not incorrectly describe it as the node-classification budget. The current repository implementation changed after the 2024 paper; pinning is required and version differences must be visible.

### Dependencies

`setup.py` requires Python>=3.10, explicitly classifies **3.11 and3.12**, and requires Torch>=2.4, PyG>=2.6, NetworkX>=2.7 plus NumPy/SciPy/scikit-learn. README249–251 and `docs/source/notes/installation.rst` describe a current **CPU reference environment Python3.11/3.12, Torch2.11.0, PyG2.8.0**. Current author CI exercises CPU on Ubuntu3.11/3.12; the PDF's historical CI claims Windows3.7–3.9, not a current Windows3.12 run. Do not claim the newer CPU reference combination has been validated on this Windows host. The minimums allow a supported Torch2.4/PyG2.6 combination, but its actual installation and native behavior are unverified.

## Additional screened exclusions and unresolved alternatives

| Paper/repository | Evidence and reason to exclude or defer |
| --- | --- |
| GKAN, arXiv2406.06470 | PDF p.2 names `https://github.com/ANRGUSC/GKAN`, which is currently unavailable. Alternate `graphkolarnoldnetwork/gkan@7c2dfa17361d9cadb23f053563f12ad3dfbe712a` has an uninitialized `pygcn_all` gitlink without `.gitmodules`, and `pykan/kan/__init__.py` imports absent `.KAN`. Do not substitute incomplete alternate source for the PDF's author source. `gkan.pdf` SHA-256 `7f9c151a27b9b1c86ea74607d3646a41eecbf9e935535e4b965988dd7f925cc2`. |
| GraphKAN, WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks | CPU source exists, but its experimental README did not establish a fixed paper identity plus locatable classification result table. |
| FastKAN, ZiyaoLi/fast-kan, arXiv2405.06721 | CPU MNIST example exists; the four-page paper's only Table1 is GPU latency. Classification accuracy appears only in Figure2. It fails this stage's classification-result-table gate; do not pair accuracy with latency. |
| Dir-GNN, emalgorithm/directed-graph-neural-network@7ba9255c53ff7bc6adc9ae24e5fc2648f29167c8 | README has CPU installation, but `src/run.py:72` always passes `devices=[args.gpu_idx]` even when `src/utils/utils.py` returns CPU. Lightning2.0.2's CPU devices must be an integer, so the full CPU entrypoint has a static incompatibility. No repair was attempted. |
| MaxCutPool, NGMLGroup/MaxCutPool@109b50b929bcd42c8b7a4bf4045d32fe28e7bf7d | Full published `run_graph_classification.py:119` hardcodes accelerator='gpu' and contains an add-non-GPU-case TODO. A CPU minimal example is not a substitute for its full published experiment runner. README fixes Python3.10/CUDA. |
| KANbeFair, yu-rp/KANbeFair, arXiv2407.16674 | A real CPU branch exists; do not call it hardcoded CUDA. Official environment pins Python3.9.7/Torch2.2.2/torchtext0.17.2 and old sklearn1.1.3. Modern3.11/3.12 compatibility remains unresolved; it is not being counted to force three candidates. |
| ProG, sheldonresearch/ProG@e0b8021e2d23353049c2af2946082d41f36a1034 | Current ProG-V2 rewrite advertises Python3.11/CPU and full200-epoch scripts, but mapping the NeurIPS2024 paper's identity, exact original protocol and result table to this rewrite remains unresolved. README smoke commands are not acceptable substitutes. Not counted as a qualified candidate. |

Together with Polynormer there are now three **statically screened** options, with all limitations stated before training. Parent must decide and fix any next-batch order before author training, register identities without overwriting retired historical papers, rerun current-source freeze/CI, and accept only actual browser/API evidence with required independent/comparison gates. Product repairs caused by any candidate retire that same paper to regression; these reports never constitute an independent pass. Keep `paper_metrics_verified=false` and `is_reproduced=null`.
