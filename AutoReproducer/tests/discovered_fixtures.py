"""Invented papers and two unrelated native scripts; never real holdout data."""
from copy import deepcopy
import hashlib
import json
import re

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
        if kwargs["task"] == "repository_identity_review":
            view = json.loads(prompt.split("Fetched evidence:\n", 1)[1])
            return json.dumps(identity_review(view))
        if kwargs['task'] == 'experiment_protocol':
            return json.dumps(fixture_protocol(self.proposal, json.loads(prompt.split('Protocol evidence:\n', 1)[1])))
        if kwargs["task"] == "discovered_repository_plan_review":
            return json.dumps(self.review)
        return json.dumps(self.prefixes.pop(0) if self.prefixes else self.proposal)


def fixture_protocol(proposal, view):
    """Invented protocol data only tests the evidence and binding mechanics."""
    excerpts = view['evidence_excerpts']
    selected = {}
    def selected_id(identifier):
        if identifier in selected:
            return selected[identifier]
        citation = next(c for c in proposal['citations'] if c['id']==identifier)
        if 'evidence_id' in citation:
            selected[citation['id']] = citation['evidence_id']
        else:
            selected[citation['id']] = next(key for key, excerpt in excerpts.items()
                if excerpt['source'] == citation['source'] and re.sub(r'\s+', ' ', citation['quote']).strip()
                    in re.sub(r'\s+', ' ', excerpt['quote']).strip()
                and excerpt.get('page') == citation.get('page') and excerpt.get('path') == citation.get('path'))
        return selected[identifier]
    def refs(ids):
        return list(dict.fromkeys(selected_id(key) for key in ids))
    both = refs(['paper_protocol', 'readme', 'script'])
    step = next(step for step in proposal['steps'] if step['kind']=='train')
    count = proposal['capture'].get('expected_runs', 1)
    return {'version':1, 'method':{'name':'Fixture Linear','variant':'Fixture','citations':both},
        'dataset':{'name':'Fixture Grid','split':'Full fixture test split','preprocessing':'Author fixture tensors','citations':both},
        'training':{'description':'Full author fixture optimization', 'parameters':[
            {'name':key,'value':value,'citations':refs(proposal['parameter_citations'][key])}
            for key, value in proposal['parameters'].items()], 'citations':both},
        'repetitions':{'count':count, 'aggregation':'mean' if count>1 else 'single','citations':both},
        'metrics':[{**{k:m[k] for k in ('name','reference','unit')}, 'table_context':'Fixture row and test column',
                    'citations':refs(m['citations'])} for m in proposal['metrics']],
        'implementation':{'argv':step['argv'],'cwd':step['cwd'],'device':proposal['capture'].get('device','cpu'),
                          'citations':refs(step['citations'])},
        'alignment':{'status':'matched','reason':'Invented fixture sources agree on the selected full protocol.', 'citations':both}}


def identity_review(view):
    """Fixture verdict: verify citation mechanics, not real-world authorship."""
    if "text" not in view["pdf"]["pages"][0]:
        excerpts = view["evidence_excerpts"]
        pdf_id = next(key for key in excerpts if key.startswith("pdf_p"))
        repo_id = next(key for key in excerpts if key.startswith("repo_"))
        review = {"verdict": "author_repository", "reason": "Invented fixture sources describe the same author experiment.",
                "checks": {gate: {"passed": True, "reason": "Fixture paper and pinned README agree.",
                    "citations": ["pdf_link_0" if gate == "author_implementation" else pdf_id, repo_id]}
                    for gate in ("paper_identity", "author_implementation", "method_alignment")}}
        if view.get("repository_input", {}).get("source") == "user_supplied":
            review["verdict"] = "matching_repository"
            review["checks"]["author_implementation"] = {"passed": False,
                "reason": "The invented fixture has no independent author ownership evidence.", "citations": []}
        return review
    pdf = view["pdf"]["pages"][0]
    link = view["pdf"]["repository_links"][0]
    path, readme = next(iter(view["repository"]["files"].items()))
    repository_quote = readme["text"].splitlines()[0]
    return {"verdict": "author_repository", "reason": "Invented fixture sources describe the same author experiment.",
            "checks": {gate: {"passed": True, "reason": "Fixture paper and actual pinned README agree.",
                "citations": [{"source": "pdf", "page": link["page"] if gate == "author_implementation" else pdf["page"],
                               "quote": link["context"] if gate == "author_implementation" else pdf["text"].splitlines()[0]},
                              {"source": "repository", "path": path, "quote": repository_quote}]}
                       for gate in ("paper_identity", "author_implementation", "method_alignment")}}
