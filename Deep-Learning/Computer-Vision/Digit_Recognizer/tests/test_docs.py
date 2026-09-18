"""The documentation is checked against the code it describes.

The README, the data notes and the CI workflow all quote commands,
flags, file paths and counts. None of that is executed anywhere, so it rots the
moment a flag is renamed. These tests turn that rot into a test failure.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
from functools import cache
from pathlib import Path

import pytest

from src.experiments import FACTORS
from src.utils import PROJECT_ROOT

README = PROJECT_ROOT / "README.md"
DATA_NOTES = PROJECT_ROOT / "data" / "README.md"
MAKEFILE = PROJECT_ROOT / "Makefile"
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "check.yml"

ANALYSIS = PROJECT_ROOT / "Mathematical_theoretical_analysis.md"

# Only documents the repository actually ships. Everything else at the root is
# a personal note, gitignored, and simply not there when CI clones the repo.
DOCS = (README, DATA_NOTES, ANALYSIS)

# Which module each modelling target runs, for checking the flags quoted in docs.
TARGET_MODULES = {
    "train": "src.train",
    "experiments": "src.experiments",
    "evaluate": "src.evaluate",
    "ensemble": "src.ensemble",
    "compare-cnn": "src.reference_cnn",
}


def makefile_targets() -> set[str]:
    """Return every target defined in the Makefile."""
    return set(re.findall(r"^([a-z][a-z0-9-]*):", MAKEFILE.read_text(), flags=re.MULTILINE))


def documented_targets() -> set[str]:
    """Return the targets that `make help` lists, from their `## name:` lines."""
    return set(re.findall(r"^## ([a-z][a-z0-9-]*):", MAKEFILE.read_text(), flags=re.MULTILINE))


def joined_lines(text: str) -> list[str]:
    """Return the text's lines with backslash continuations folded into one."""
    return re.sub(r"\\\n\s*", " ", text).splitlines()


def code_only(document: Path) -> str:
    """Return just the fenced blocks and inline spans of a markdown file.

    Prose says things like "make sure"; only code is a claim about the project.
    """
    text = document.read_text()
    blocks = re.findall(r"```[a-z]*\n(.*?)```", text, flags=re.DOTALL)
    spans = re.findall(r"`([^`\n]+)`", text)
    return "\n".join(blocks + spans)


def parser_flags(module_name: str) -> set[str]:
    """Return every long flag the module's argument parser accepts."""
    module = importlib.import_module(module_name)
    builder = getattr(module, "build_parser", None)
    if builder is None:  # src.evaluate builds its parser inside main()
        source = Path(module.__file__ or "").read_text()
        return set(re.findall(r'add_argument\(\s*"(--[a-z-]+)"', source))
    parser: argparse.ArgumentParser = builder()
    return {option for action in parser._actions for option in action.option_strings}


@pytest.mark.parametrize("document", DOCS, ids=lambda path: path.name)
def test_every_make_target_the_docs_mention_exists(document: Path) -> None:
    targets = makefile_targets()
    for target in set(re.findall(r"\bmake ([a-z][a-z0-9-]*)", code_only(document))):
        assert target in targets, f"{document.name} documents `make {target}`, which does not exist"


def test_every_make_target_is_listed_by_make_help() -> None:
    """A target missing its `## ` line is invisible to anyone reading the help."""
    internal = {"help"}
    missing = makefile_targets() - documented_targets() - internal
    assert not missing, f"targets with no `## ` help line: {sorted(missing)}"


@pytest.mark.parametrize("document", DOCS, ids=lambda path: path.name)
def test_every_module_the_docs_invoke_is_runnable(document: Path) -> None:
    for module_name in set(re.findall(r"python -m (src\.[a-z_]+)", code_only(document))):
        module = importlib.import_module(module_name)
        assert hasattr(module, "main"), f"{module_name} has no main() to run"


@pytest.mark.parametrize("document", DOCS, ids=lambda path: path.name)
def test_every_flag_quoted_in_the_docs_is_accepted_by_its_parser(document: Path) -> None:
    """A renamed flag must fail here, not when the command is copied out."""
    for line in joined_lines(document.read_text()):
        if match := re.search(r"python -m (src\.[a-z_]+)(.*)", line):
            module_name, rest = match.group(1), match.group(2)
        elif match := re.search(r"make ([a-z-]+) ARGS=\"(.*?)\"", line):
            target, rest = match.group(1), match.group(2)
            module_name = TARGET_MODULES.get(target, "")
            if not module_name:
                continue
        else:
            continue

        accepted = parser_flags(module_name)
        for flag in re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", rest):
            assert flag in accepted, f"{document.name}: {module_name} has no {flag}"


@pytest.mark.parametrize("document", DOCS, ids=lambda path: path.name)
def test_every_factor_named_in_the_docs_is_a_real_factor(document: Path) -> None:
    """`--factors` takes field names; a typo would silently sweep nothing."""
    for line in joined_lines(document.read_text()):
        if match := re.search(r"--factors ([a-z_ ]+)", line):
            for factor in match.group(1).split():
                assert factor in FACTORS, f"{document.name} sweeps unknown factor {factor!r}"


def test_the_chunked_sweep_in_the_readme_covers_every_factor() -> None:
    """The chunks are offered as a way to run the whole sweep in pieces."""
    chunked = {
        factor
        for line in joined_lines(README.read_text())
        if "--out reports/experiments_part" in line
        for factor in re.search(r"--factors ([a-z_ ]+)", line).group(1).split()  # type: ignore[union-attr]
    }
    assert chunked == set(FACTORS), f"chunks miss {set(FACTORS) - chunked}"


def test_the_sweep_counts_in_the_readme_match_the_factor_table() -> None:
    levels = sum(len(values) for values in FACTORS.values())
    text = README.read_text()
    assert f"{len(FACTORS)} factors, {levels} levels" in text
    assert f"{len(FACTORS)} factors" in text


# Directories that live under the root but are not the project. They are
# skipped when the tree below is checked, for two reasons. Speed: .venv alone
# holds forty thousand files, and a glob per tree entry re-walked all of them.
# Honesty: .venv ships modules called utils.py, config.py and __init__.py, so
# a search of the whole root finds a match for a src/ module that has in fact
# been deleted.
NOT_THE_PROJECT = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "lightning_logs",
}


@cache
def project_file_names() -> frozenset[str]:
    """Every file name the project itself contains, from one walk of the root."""
    names: set[str] = set()
    for _directory, subdirectories, files in os.walk(PROJECT_ROOT):
        subdirectories[:] = [name for name in subdirectories if name not in NOT_THE_PROJECT]
        names.update(files)
    return frozenset(names)


def test_every_path_in_the_project_tree_exists() -> None:
    """The README tree is a map; a stale entry sends the reader nowhere."""
    tree = README.read_text().split("## Project structure")[1].split("```")[1]
    written = project_file_names()
    for name in re.findall(r"[├└]── ([\w./-]+)", tree):
        if name.endswith("/"):
            continue
        # A bare file name may sit anywhere; one with a directory in it is
        # a promise about where, so check it there.
        found = (PROJECT_ROOT / name).exists() if "/" in name else name in written
        assert found, f"the project tree lists {name}, which does not exist"


def test_every_source_and_test_module_appears_in_the_project_tree() -> None:
    """The other direction: a new module must be added to the map."""
    tree = README.read_text().split("## Project structure")[1].split("```")[1]
    for path in sorted(PROJECT_ROOT.glob("src/*.py")) + sorted(PROJECT_ROOT.glob("tests/*.py")):
        assert path.name in tree, f"{path.name} is missing from the README project tree"


def test_claude_md_is_left_out_of_the_project_tree() -> None:
    """It is not pushed, so documenting it would describe a file nobody has."""
    tree = README.read_text().split("## Project structure")[1].split("```")[1]
    assert "CLAUDE.md" not in tree


def test_the_workflow_runs_the_same_gate_as_make_check() -> None:
    """CI has to be the local gate, or a green tick means nothing."""
    workflow = WORKFLOW.read_text()
    check_line = re.search(r"^check:(.*)$", MAKEFILE.read_text(), flags=re.MULTILINE)
    assert check_line is not None
    for target in check_line.group(1).split():
        assert f"make {target}" in workflow, f"CI does not run `make {target}`"


def test_the_workflow_installs_the_pinned_dependencies() -> None:
    """`sync-ci` uses --frozen, so CI tests what uv.lock actually pins."""
    assert "make sync-ci" in WORKFLOW.read_text()
    assert "--frozen" in MAKEFILE.read_text()


def test_the_documented_kaggle_input_path_is_used_consistently() -> None:
    """One path is quoted in several places; they must agree."""
    text = README.read_text()
    paths = set(re.findall(r"(/kaggle/input/digit-recognizer/[\w.]+)", text))
    assert paths == {"/kaggle/input/digit-recognizer/train.csv"}


# GitHub renders $...$ and $$...$$ with a KaTeX subset and refuses these
# outright, printing "The following macros are not allowed" in place of the
# formula. They render fine in Jupyter and in most local previews, which is
# why the difference only shows up once something is pushed.
# One backslash, not two: these are substring checks against the file, and a
# document holds `\operatorname`, so the doubled form matched nothing.
MACROS_GITHUB_REFUSES = (
    r"\operatorname",
    r"\DeclareMathOperator",
    r"\newcommand",
    "\\def\\",
    r"\phantom",
    r"\hphantom",
    r"\vphantom",
)

# GitHub runs its markdown escape pass over the math before KaTeX sees it, so a
# `\_` written inside \text{...} arrives at the renderer as a bare underscore
# and is refused with "'_' allowed only in math mode". Escaping is not the fix,
# because the escape is what gets eaten: the identifier has to leave math mode
# altogether and live in an inline code span, where underscores are ordinary.
TEXT_MODE_UNDERSCORE = re.compile(
    r"\\(?:text|textbf|textit|texttt|mathrm|mathbf|mathit)\{[^}]*_[^}]*\}"
)


@pytest.mark.parametrize("macro", MACROS_GITHUB_REFUSES)
def test_no_document_uses_a_macro_github_refuses(macro: str) -> None:
    """A rejected macro replaces the equation with an error message on GitHub."""
    notebook = PROJECT_ROOT / "notebooks" / "mnist_mlp_report.ipynb"
    for path in (*DOCS, notebook):
        assert macro not in path.read_text(), (
            f"{path.name} uses {macro}, which GitHub will not render. "
            r"Use \mathrm{...} for upright operator names."
        )


def test_no_document_puts_an_underscore_in_math_text_mode() -> None:
    """An identifier like sgd_momentum belongs in a code span, not in \\textbf."""
    notebook = PROJECT_ROOT / "notebooks" / "mnist_mlp_report.ipynb"
    for path in (*DOCS, notebook):
        found = TEXT_MODE_UNDERSCORE.findall(path.read_text())
        assert not found, (
            f"{path.name} has {found[0]!r} in math text mode. GitHub strips the "
            "backslash from an escaped underscore before KaTeX runs, so the "
            "equation is replaced by an error. Put the identifier in an inline "
            "code span outside the math instead."
        )
