"""Invented papers and two unrelated native scripts; never real holdout data."""
from copy import deepcopy
import hashlib
import json

from PyPDF2 import PdfWriter
from PyPDF2.generic import DecodedStreamObject, DictionaryObject, NameObject


def write_pdf(path, pages):
    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    for lines in pages:
        writer.add_blank_page(width=612, height=792)
        page = writer.pages[-1]
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        commands = ["BT /F1 12 Tf 72 720 Td"]
        for index, line in enumerate(lines):
            if index:
                commands.append("0 -18 Td")
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"({escaped}) Tj")
        commands.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(commands).encode("ascii"))
        page[NameObject("/Contents")] = stream
    with path.open("wb") as output:
        writer.write(output)
    return path


def make_experiment(root, family="signals"):
    """Return actual PDF, immutable source snapshot and an evidence proposal."""
    workspace = root / ("author_" + family)
    workspace.mkdir(parents=True)
    paired = family == "pairs"
    epochs, samples = (60, 8) if paired else (40, 4)
    title = "Paired Sensor Classifiers" if paired else "Separating Signed Signals"
    method = "Paired Linear" if paired else "Signal Linear"
    dataset = "PairGrid" if paired else "SignGrid"
    script_name = "fit_pairs.py" if paired else "train.py"
    url = "https://github.com/invented-fixtures/" + family
    repo = {"url": url, "revision": "a" * 40}
    paper_lines = [title, "Abstract", "Our source code is available at " + url,
                   f"{method} uses the complete {dataset} dataset and fixed test split."]
    metrics_lines = ["Evaluation protocol and results", "Method Dataset Split Accuracy (percent)",
                     f"{method} {dataset} test 100.0", f"Baseline {dataset} train 98.0",
                     f"Use {epochs} epochs, seed 7, and all {samples} test samples."]
    pdf = write_pdf(root / (family + ".pdf"), [paper_lines, ["Full methods on the next page."], metrics_lines])
    readme = (f"{title}\nMethod: {method}; dataset: {dataset}; fixed complete test split.\n"
              f"Run python {script_name} --epochs {epochs} --seed 7 --cpu\n"
              f"Train for {epochs} optimizer steps and evaluate all {samples} test samples.\n"
              "All data is in dataset.json; no external or generated input is used.\n")
    common = ("import argparse\nimport json\nimport torch\n"
              "parser = argparse.ArgumentParser()\n"
              f"parser.add_argument('--epochs', type=int, default={epochs})\n"
              "parser.add_argument('--seed', type=int, default=7)\n"
              "parser.add_argument('--cpu', action='store_true')\n"
              "args = parser.parse_args()\ntorch.manual_seed(args.seed)\n"
              "with open('dataset.json', encoding='utf-8') as stream:\n"
              "    records = json.load(stream)\n")
    if paired:
        records = {"left": [[-2.], [-1.], [1.], [2.]] * 4,
                   "right": [[-1.], [-2.], [2.], [1.]] * 4,
                   "labels": [0, 0, 1, 1] * 4}
        model_source = ("import torch\nclass PairModel(torch.nn.Module):\n"
                        "    def __init__(self):\n"
                        "        super().__init__()\n"
                        "        self.linear = torch.nn.Linear(2, 2)\n"
                        "    def forward(self, left, right):\n"
                        "        return self.linear(torch.cat([left, right], dim=1))\n")
        body = ("from author_model import PairModel\n"
                "left_all = torch.tensor(records['left'], dtype=torch.float32)\n"
                "right_all = torch.tensor(records['right'], dtype=torch.float32)\n"
                "all_labels = torch.tensor(records['labels'], dtype=torch.long)\n"
                "model = PairModel()\noptimizer = torch.optim.SGD(model.parameters(), lr=0.1)\n"
                "for epoch in range(args.epochs):\n"
                "    optimizer.zero_grad()\n"
                "    loss = torch.nn.functional.cross_entropy(model(left_all[:8], right_all[:8]), all_labels[:8])\n"
                "    loss.backward()\n    optimizer.step()\n"
                "left = left_all[8:]\nright = right_all[8:]\nlabels = all_labels[8:]\n"
                "model.eval()\nwith torch.no_grad():\n"
                "    logits = model(left, right)\n"
                "    accuracy = (logits.argmax(1) == labels).float().mean().item()\n"
                "print('Test Accuracy:', accuracy)\n")
        forward_args, test_indices = ["left", "right"], None
    else:
        records = {"features": [[-2.], [-1.], [1.], [2.]] * 3, "labels": [0, 0, 1, 1] * 3}
        model_source = ("import torch\nclass SignalModel(torch.nn.Module):\n"
                        "    def __init__(self):\n"
                        "        super().__init__()\n"
                        "        self.linear = torch.nn.Linear(1, 2)\n"
                        "    def forward(self, features):\n"
                        "        return self.linear(features)\n")
        body = ("from author_model import SignalModel\n"
                "features = torch.tensor(records['features'], dtype=torch.float32)\n"
                "labels = torch.tensor(records['labels'], dtype=torch.long)\n"
                "idx_test = torch.tensor([8, 9, 10, 11])\n"
                "model = SignalModel()\n"
                "optimizer = torch.optim.SGD(model.parameters(), lr=0.1)\n"
                "for epoch in range(args.epochs):\n"
                "    optimizer.zero_grad()\n"
                "    loss = torch.nn.functional.cross_entropy(model(features[:8]), labels[:8])\n"
                "    loss.backward()\n    optimizer.step()\n"
                "model.eval()\nwith torch.no_grad():\n"
                "    logits = model(features)\n"
                "    accuracy = (logits[idx_test].argmax(1) == labels[idx_test]).float().mean().item()\n"
                "print('Test Accuracy:', accuracy)\n")
        forward_args, test_indices = ["features"], "idx_test"
    sources = {"README.md": readme, "requirements.txt": "torch==1.13.1\n",
               script_name: common + body, "author_model.py": model_source, "dataset.json": json.dumps(records)}
    snapshot = {**repo, "files": {}}
    for name, text in sources.items():
        raw = text.encode("utf-8")
        (workspace / name).write_bytes(raw)
        snapshot["files"][name] = hashlib.sha256(raw).hexdigest()
    citations = [
        {"id": "paper_protocol", "source": "pdf", "page": 1, "quote": paper_lines[-1]},
        {"id": "paper_results", "source": "pdf", "page": 3, "quote": "\n".join(metrics_lines)},
        {"id": "readme", "source": "repository", "path": "README.md", "quote": readme},
        {"id": "script", "source": "repository", "path": script_name, "quote": sources[script_name]},
        {"id": "model", "source": "repository", "path": "author_model.py", "quote": model_source},
        {"id": "requirements", "source": "repository", "path": "requirements.txt", "quote": "torch==1.13.1"},
    ]
    proposal = {
        "version": 1, "scope": "selected_paper_experiment", "repository": repo,
        "citations": citations,
        "claims": {"protocol": ["paper_protocol", "readme", "script"],
                   "dataset": ["paper_protocol", "readme", "script"], "metrics": ["paper_results"]},
        "parameters": {"epochs": epochs, "seed": 7},
        "parameter_citations": {"epochs": ["script", "paper_results"], "seed": ["script", "paper_results"]},
        "requirements": {"author": ["torch==1.13.1"],
                         "compatibility": ["torch==2.5.1+cpu", "numpy==1.26.4"],
                         "reason": "Basic tensor and optimizer APIs remain compatible; NumPy is evaluator-only.",
                         "citations": ["requirements", "script"]},
        "datasets": [{"kind": "bundled", "paths": ["dataset.json"], "citations": ["readme", "script"]}],
        "steps": [{"id": "train", "kind": "train",
                   "argv": ["python", script_name, "--epochs", str(epochs), "--seed", "7", "--cpu"],
                   "cwd": ".", "timeout_s": 120, "citations": ["readme", "script"]}],
        "metrics": [{"name": "accuracy", "reference": 100.0, "unit": "percent", "direction": "maximize",
                     "tolerance_relative": 0.05, "citations": ["paper_results"]}],
        "capture": {"kind": "torch_classification", "model": "model", "forward_args": forward_args,
                    "labels": "labels", "test_indices": test_indices,
                    "expected_optimizer_steps": epochs, "expected_test_samples": samples,
                    "reported_metric": {"label": "Test Accuracy", "unit": "fraction"},
                    "citations": ["readme", "script"]},
    }
    return {"pdf": pdf, "workspace": workspace, "snapshot": snapshot, "proposal": proposal}


def accepted_review():
    """Mock the network response only, never claim fixture semantic inference."""
    from src.discovered_repository_plan import _SEMANTIC_GATES
    return {"accepted": True, "checks": {gate: {
        "passed": True, "reason": "Fixture evidence matches this fixed complete experiment.",
        "citations": ["paper_protocol", "paper_results", "readme", "script", "requirements"],
    } for gate in _SEMANTIC_GATES}}


class PlannedLLM:
    def __init__(self, proposal, review=None, prefixes=()):
        self.proposal, self.review = deepcopy(proposal), deepcopy(review if review is not None else accepted_review())
        self.prefixes, self.calls = list(prefixes), []

    def get_call_count(self):
        return len(self.calls)

    def chat(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if kwargs["task"] == "discovered_repository_plan_review":
            return json.dumps(self.review)
        return json.dumps(self.prefixes.pop(0) if self.prefixes else self.proposal)
