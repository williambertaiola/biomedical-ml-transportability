"""Publication hygiene and configurable-path regression tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from thesis_tools.paths import REPOSITORY_ROOT, configured_path, resolve_config_paths


def test_paths_resolve_without_working_directory_dependency(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Path(configured_path("data_root")) == REPOSITORY_ROOT / "data"
    assert Path(configured_path("results_root")) == REPOSITORY_ROOT / "results"
    external = tmp_path / "private_inputs"
    monkeypatch.setenv("THESIS_DATA_ROOT", str(external))
    result = resolve_config_paths(
        {"input_paths": ["data/raw/uci/cleveland.data"], "random_seed": 20260327}
    )
    assert result["input_paths"] == [str(external / "raw/uci/cleveland.data")]
    assert result["random_seed"] == 20260327


def test_all_source_files_compile_and_have_no_personal_paths():
    for source in REPOSITORY_ROOT.rglob("*.py"):
        if ".venv" in source.parts or "venv" in source.parts:
            continue
        text = source.read_text(encoding="utf-8")
        compile(text, str(source), "exec")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not (
                    node.value.startswith(("/Users/", "/home/"))
                    and len(Path(node.value).parts) > 2
                ), source


def test_all_original_python_files_accounted_for():
    manifest = json.loads(
        (REPOSITORY_ROOT / "docs" / "code_provenance.json").read_text()
    )
    assert len(manifest) == 495
    assert len({item["original_sha256"] for item in manifest}) == 255
    for item in manifest:
        assert (REPOSITORY_ROOT / item["final_path"]).is_file(), item["original_path"]


def test_public_repository_has_no_data_or_model_artifacts():
    # The ignored data/results roots are intentionally allowed for local work.
    for path in REPOSITORY_ROOT.rglob("*"):
        relative = path.relative_to(REPOSITORY_ROOT)
        if relative.parts[0] in {"data", "results", ".venv", "venv"}:
            continue
        if path.is_file():
            assert path.suffix not in {
                ".csv",
                ".tsv",
                ".xlsx",
                ".joblib",
                ".pkl",
                ".rds",
                ".data",
                ".zip",
            }, path


def test_core_scientific_definitions_match_original_ast():
    """Protect model, split and method definitions against unintended edits."""
    import copy
    import hashlib

    def fingerprint(node):
        node = copy.deepcopy(node)
        for part in ast.walk(node):
            if (
                isinstance(part, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and part.body
            ):
                first = part.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    part.body.pop(0)

        def stable(part):
            if isinstance(part, ast.AST):
                return [
                    type(part).__name__,
                    {
                        key: stable(value)
                        for key, value in ast.iter_fields(part)
                        if value is not None and value != []
                    },
                ]
            if isinstance(part, list):
                return [stable(value) for value in part]
            return repr(part)

        serialized = json.dumps(
            stable(node), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(serialized).hexdigest()

    manifest = json.loads((REPOSITORY_ROOT / "docs/code_provenance.json").read_text())
    shared = {
        node.name: node
        for node in ast.parse(
            (REPOSITORY_ROOT / "src/thesis_tools/metrics.py").read_text()
        ).body
        if isinstance(node, ast.FunctionDef)
    }
    for item in manifest:
        if not item["final_path"].startswith(
            (
                "src/uci_transportability/",
                "src/abide_transportability/",
                "src/tcga_transportability/",
            )
        ):
            continue
        nodes = {
            node.name: node
            for node in ast.parse(
                (REPOSITORY_ROOT / item["final_path"]).read_text()
            ).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        for name, expected in item["functions"].items():
            node = nodes.get(name, shared.get(name))
            assert node is not None, (item["final_path"], name)
            assert fingerprint(node) == expected, (item["final_path"], name)
