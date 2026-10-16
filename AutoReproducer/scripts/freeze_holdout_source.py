"""Record each product generation before its independent author-code execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
args = parser.parse_args()
files = [root / "app.py"]
for directory in ("src", "frontend"):
    files.extend(path for path in (root / directory).rglob("*.py") if "__pycache__" not in path.parts)
hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
          for path in sorted(files)}
record = {"frozen_at": datetime.now(timezone.utc).isoformat(),
          "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
          "files": hashes,
          "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
          "excluded_regression_papers": ["DeepXDE", "ReZero", "DLinear", "Neural Ordinary Differential Equations", "SIREN", "Simplifying Graph Convolutional Networks", "Simple Spectral Graph Convolution", "Adaptive Universal Generalized PageRank Graph Neural Network", "Improving Graph Neural Networks with Simple Architecture Design", "Simple and Deep Graph Convolutional Networks", "Interpreting and Unifying Graph Neural Networks with An Optimization Framework", "DropEdge: Towards Deep Graph Convolutional Networks on Node Classification", "Revisiting Heterophily for Graph Neural Networks", "Two Sides of the Same Coin: Heterophily and Oversmoothing in Graph Convolutional Neural Networks"],
          "rule": "Any product repair after a PDF trial retires that paper; choose another PDF after the next freeze."}
record['objective'] = 'author_code_execution'
record['candidate_acquisition_rule'] = 'User explicitly allowed preparing 10 candidates; advance PDF acquisition is allowed. No retired repair paper may be reused for final acceptance.'
record['excluded_regression_papers'].append('How Powerful are Spectral Graph Neural Networks')
record['excluded_regression_papers'].append('Towards Deeper Graph Neural Networks')
record['excluded_regression_papers'].append('Diffusion Improves Graph Learning')
record['excluded_regression_papers'].append('Graph Structure Learning for Robust Graph Neural Networks')
record['excluded_regression_papers'].append('Dissecting the Diffusion Process in Linear Graph Convolutional Networks')
record['excluded_regression_papers'].append('EvenNet: Ignoring Odd-Hop Neighbors Improves Robustness of Graph Neural Networks')
record['excluded_regression_papers'].append('Hyperbolic Graph Convolutional Neural Networks')
record['excluded_regression_papers'].extend(['Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited',
    'BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation', 'Deep Graph Infomax',
    'PairNorm: Tackling Oversmoothing in GNNs', 'How Powerful Are Graph Neural Networks?',
    'Graph Random Neural Networks for Semi-Supervised Learning on Graphs'])
record['excluded_regression_papers'].extend(['Graph Contrastive Learning with Adaptive Augmentation',
    'Deep Graph Contrastive Representation Learning',
    'Large-Scale Representation Learning on Graphs via Bootstrapping',
    'How Attentive are Graph Attention Networks?',
    'Graph Neural Networks with Adaptive Residual',
    'Elastic Graph Neural Networks',
    'Graph Neural Networks with Adaptive Frequency Response Filter',
    'GMNN: Graph Markov Neural Networks',
    'Beyond Low-frequency Information in Graph Convolutional Networks',
    'Predict then Propagate: Graph Neural Networks meet Personalized PageRank',
    'Contrastive Multi-View Representation Learning on Graphs', 'Graph U-Nets',
    'SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations',
    'Finding Global Homophily in Graph Neural Networks When Meeting Heterophily'])
record['excluded_regression_papers'].append('Combining Label Propagation and Simple Models Out-performs Graph Neural Networks')
record['excluded_regression_papers'].append('NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classification')
target = root / "data/runtime" / (args.tag + "_source_freeze.json")
if target.exists():
    raise ValueError("Never overwrite a previous holdout source freeze")
target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in record.items() if key != "files"}, ensure_ascii=False))
print(target)
