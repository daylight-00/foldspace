"""The installable package holds what the code imports.

The wheel is built from `src/foldspace`, so every `foldspace.*` module that code in this repository imports has to be a file
under it, and every directory of it that holds modules has to be a package. A rename or a move that left an import behind
would pass on a checkout, where `experiments/` and `tests/` sit beside `src/`, and fail on an installed package. Standard
library only.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.offline

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _python_files() -> list[Path]:
    found = []
    for top in ("src", "tests", "experiments"):
        found += [p for p in (ROOT / top).rglob("*.py") if "__pycache__" not in p.parts]
    return found


def test_the_wheel_is_built_from_the_one_package_that_holds_the_code() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["src/foldspace"]
    assert {p.name for p in SRC.iterdir() if p.is_dir()} == {"foldspace"}


def test_every_directory_of_the_package_that_holds_modules_is_a_package() -> None:
    missing = [
        str(d.relative_to(ROOT))
        for d in (SRC / "foldspace").rglob("*")
        if d.is_dir()
        and "__pycache__" not in d.parts
        and any(d.glob("*.py"))
        and not (d / "__init__.py").exists()
    ]
    assert not missing, missing


def test_every_import_of_the_package_resolves_to_a_module_of_it() -> None:
    unresolved = []
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
                parts = node.module.split(".")
                modules += [".".join([*parts, a.name]) for a in node.names]
            elif isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            else:
                continue
            for name in modules:
                if name.split(".")[0] != "foldspace":
                    continue
                target = SRC.joinpath(*name.split("."))
                # a name imported from a module is an attribute, not a file: only the module itself must exist
                if name == node.module if isinstance(node, ast.ImportFrom) else True:
                    if not (target.with_suffix(".py").exists() or (target / "__init__.py").exists()):
                        unresolved.append(f"{path.relative_to(ROOT)}: {name}")
    assert not unresolved, unresolved


def test_every_command_of_the_documents_names_a_module() -> None:
    """The documented commands run modules by name (`python -m foldspace.<package>.<module>`); each has to exist."""
    import re

    missing = []
    documents = sorted((ROOT / "reproducibility").glob("*.md")) + sorted((ROOT / "docs").glob("*.md"))
    for document in documents:
        # a placeholder such as `foldspace.rfd4.<module>` is not a command
        for name in re.findall(r"python -m (foldspace(?:\.\w+)+)(?![\w.<])", document.read_text()):
            if not SRC.joinpath(*name.split(".")).with_suffix(".py").exists():
                missing.append(f"{document.relative_to(ROOT)}: {name}")
    assert not missing, missing


# ---------------------------------------------------------------------------------------- documented command lines
def _parsers(path: Path) -> dict:
    """The options of each parser of a module, read from its source: {subcommand or None: [(flags, required), ...]}."""
    tree = ast.parse(path.read_text())
    owner: dict[str, str | None] = {}
    model: dict[str | None, list] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and len(node.targets) == 1):
            continue
        func, target = node.value.func, node.targets[0]
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if not isinstance(target, ast.Name):
            continue
        if name == "ArgumentParser":
            owner[target.id] = None
            model.setdefault(None, [])
        elif name == "add_parser" and node.value.args and isinstance(node.value.args[0], ast.Constant):
            owner[target.id] = node.value.args[0].value
            model.setdefault(node.value.args[0].value, [])
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        target = node.func.value
        if not (isinstance(target, ast.Name) and target.id in owner):
            continue
        flags = tuple(a.value for a in node.args if isinstance(a, ast.Constant) and str(a.value).startswith("--"))
        required = any(
            k.arg == "required" and isinstance(k.value, ast.Constant) and k.value.value is True for k in node.keywords
        )
        if flags:
            model[owner[target.id]].append((flags, required))
    return model


def _documented_commands(text: str):
    import re

    in_block, lines = False, text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("```"):
            in_block = not in_block
        elif in_block:
            m = re.search(r"python -m (foldspace(?:\.\w+)+)(.*)", lines[i])
            if m:
                rest = m.group(2)
                while rest.rstrip().endswith("\\") and i + 1 < len(lines):
                    i += 1
                    rest = rest.rstrip()[:-1] + " " + lines[i].strip()
                yield m.group(1), re.sub(r"#.*$", "", rest).replace("[", " ").replace("]", " ").split()
        i += 1


def test_every_documented_command_gives_the_required_options_of_its_parser_and_only_known_ones() -> None:
    problems = []
    for document in sorted((ROOT / "reproducibility").glob("*.md")):
        for module, tokens in _documented_commands(document.read_text()):
            path = SRC.joinpath(*module.split(".")).with_suffix(".py")
            if not path.exists():
                continue
            model = _parsers(path)
            subcommands = [s for s in model if s]
            sub = tokens[0] if tokens and tokens[0] in subcommands else None
            if subcommands and sub is None and not model.get(None):
                problems.append(f"{document.name}: {module}: no subcommand")
                continue
            options = model.get(sub, [])
            if not options:
                continue  # a parser built in a way this reading does not follow
            given = {t.split("=")[0] for t in tokens if t.startswith("--")}
            known = {flag for flags, _ in options for flag in flags}
            for flags, required in options:
                if required and not given & set(flags):
                    problems.append(f"{document.name}: python -m {module} {sub or ''}: missing {flags[0]}")
            for flag in sorted(given - known - {"--help"}):
                problems.append(f"{document.name}: python -m {module} {sub or ''}: unknown {flag}")
    assert not problems, problems


def test_the_documented_flow_readings_open_the_trained_checkpoint_and_the_conditions_of_their_target() -> None:
    """Without `--checkpoint`, `sample_flow` and `flow_eval` read the weights of `--flow`, the pretrained flow, not the run;
    `flow_eval --target esmc` also needs `--adapter` and `--esmc`; and the scripts change into their output directory before
    they resolve the inputs, so the document has to say that paths are absolute."""
    document = ROOT / "reproducibility" / "rfd4_training.md"
    text = document.read_text()
    seen = set()
    for module, tokens in _documented_commands(text):
        name = module.rsplit(".", 1)[-1]
        if name not in ("sample_flow", "flow_eval"):
            continue
        seen.add(name)
        given = {t.split("=")[0] for t in tokens if t.startswith("--")}
        assert {"--checkpoint", "--expect-step"} <= given, f"{name} does not read the trained checkpoint"
        if name == "flow_eval" and "--target" in tokens and tokens[tokens.index("--target") + 1] == "esmc":
            assert {"--adapter", "--esmc"} <= given
    assert seen == {"sample_flow", "flow_eval"}
    assert "absolute" in text
