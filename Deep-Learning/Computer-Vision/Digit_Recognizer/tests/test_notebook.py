"""The report notebook is checked like source.

These tests guard the one failure mode a notebook has that a module does not:
it is never imported, never linted and never run by the suite, so it drifts
away from `src/` silently and nothing catches the drift.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import pytest

from src.experiments import FACTORS
from src.utils import PROJECT_ROOT

matplotlib.use("Agg")

NOTEBOOK = PROJECT_ROOT / "notebooks" / "mnist_mlp_report.ipynb"

# Literals that must never reach a committed cell. The Colab and Kaggle routes
# both involve API tokens, and a pasted one would be saved into the .ipynb.
SECRET_PATTERNS = (
    r"ghp_[A-Za-z0-9]{16,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"KAGGLE_KEY\s*=\s*['\"][^'\"]+['\"]",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
)


@pytest.fixture(scope="module")
def notebook() -> dict[str, Any]:
    """The parsed notebook."""
    parsed: dict[str, Any] = json.loads(NOTEBOOK.read_text())
    return parsed


def sources(notebook: dict[str, Any], kind: str) -> list[str]:
    """Return the joined source of every cell of one type."""
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == kind]


def strip_magics(source: str) -> str:
    """Blank out IPython magics and shell escapes so the cell parses as Python."""
    lines = source.splitlines()
    return "\n".join("" if line.lstrip().startswith(("!", "%")) else line for line in lines)


def find_cell(notebook: dict[str, Any], fragment: str) -> str:
    """Return the one cell containing a fragment, failing if it is not unique."""
    matches = [
        "".join(cell["source"]) for cell in notebook["cells"] if fragment in "".join(cell["source"])
    ]
    assert len(matches) == 1, f"expected exactly one cell with {fragment!r}, found {len(matches)}"
    return matches[0]


def test_the_notebook_is_valid_json_in_the_expected_format(notebook: dict[str, Any]) -> None:
    assert notebook["nbformat"] == 4
    assert notebook["cells"], "the notebook has no cells"


def test_every_code_cell_parses_as_python(notebook: dict[str, Any]) -> None:
    """A syntax error hides until the cell is run, which is too late."""
    for index, source in enumerate(sources(notebook, "code")):
        try:
            ast.parse(strip_magics(source))
        except SyntaxError as error:  # pragma: no cover - only on a broken notebook
            pytest.fail(f"code cell {index} does not parse: {error}")


def test_everything_imported_from_src_exists(notebook: dict[str, Any]) -> None:
    """The notebook imports from `src`; renaming a function must break a test."""
    import importlib

    for source in sources(notebook, "code"):
        for node in ast.walk(ast.parse(strip_magics(source))):
            if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("src"):
                continue
            module = importlib.import_module(node.module or "")
            for alias in node.names:
                assert hasattr(module, alias.name), f"{node.module}.{alias.name} does not exist"


def test_no_cell_contains_anything_shaped_like_a_credential(notebook: dict[str, Any]) -> None:
    """Cell outputs are committed with the notebook, so a pasted token ships."""
    blob = json.dumps(notebook)
    for pattern in SECRET_PATTERNS:
        assert not re.search(pattern, blob), f"a value matching {pattern} is in the notebook"


def test_the_sections_are_numbered_without_gaps(notebook: dict[str, Any]) -> None:
    """Headings are the table of contents in section 0; they must agree."""
    numbers = [
        int(match.group(1))
        for source in sources(notebook, "markdown")
        if (match := re.match(r"## (\d+)\. ", source))
    ]
    assert numbers == list(range(1, len(numbers) + 1))


def test_the_sweep_claims_match_the_factor_table(notebook: dict[str, Any]) -> None:
    """The prose quotes counts; `FACTORS` is where they actually come from."""
    levels = sum(len(values) for values in FACTORS.values())
    seeds = 3
    claim = find_cell(notebook, "## 4. Hyperparameter study")

    assert f"{len(FACTORS)} factors" in claim.lower().replace("ten", str(len(FACTORS)))
    assert f"{levels} levels" in claim
    assert f"{levels * seeds} runs" in claim


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_the_sweep_figure_gives_every_factor_a_panel(notebook: dict[str, Any]) -> None:
    """The bug this replaces: a hard-coded 2x4 grid silently dropped factors.

    The cell is executed here against a synthetic summary, so the test measures
    what the figure does rather than how the code happens to be written.
    """
    n_factors = len(FACTORS) + 3  # more than any fixed grid would allow for
    summary = pd.DataFrame(
        {
            "factor": [f"factor_{i}" for i in range(n_factors) for _ in range(2)],
            "level": ["a", "b"] * n_factors,
            "val_accuracy_mean": [0.9, 0.91] * n_factors,
            "val_accuracy_std": [0.01, 0.01] * n_factors,
        }
    )
    namespace: dict[str, Any] = {"summary": summary, "plt": matplotlib.pyplot}
    exec(find_cell(notebook, 'axis.set_ylabel("val accuracy")'), namespace)  # noqa: S102

    titled = [axis.get_title() for axis in namespace["figure"].get_axes() if axis.get_title()]
    assert sorted(titled) == sorted(summary["factor"].unique())
    matplotlib.pyplot.close("all")


def test_the_notebook_reads_the_test_split_in_exactly_one_place(
    notebook: dict[str, Any],
) -> None:
    """`evaluate_run` is the only path to the test labels, and it runs once."""
    calls = sum(source.count("evaluate_run(") for source in sources(notebook, "code"))
    assert calls == 1, f"evaluate_run is called {calls} times; the test split is read once"


def test_the_colab_cell_does_not_reinstall_torch(notebook: dict[str, Any]) -> None:
    """Reinstalling torch on Colab replaces the CUDA build with a CPU one."""
    installs = [
        line
        for source in sources(notebook, "code")
        for line in source.splitlines()
        if "pip" in line and "install" in line
    ]
    assert installs, "the bootstrap cell no longer installs anything"
    for line in installs:
        assert not re.search(r"\btorch\b", line), f"{line.strip()!r} reinstalls torch"


def test_the_bootstrap_detects_kaggle_before_colab(notebook: dict[str, Any]) -> None:
    """The bug this replaces: the cell asked whether `google.colab` could be
    imported, which the Kaggle image also answers yes to. Kaggle sessions took
    the Colab branch and tried to enter a /content path that is not there."""
    cell = find_cell(notebook, "def find_project")
    assert 'find_spec("google.colab")' not in cell, "that lookup succeeds on Kaggle too"
    assert "KAGGLE_KERNEL_RUN_TYPE" in cell


def test_the_bootstrap_looks_for_the_project_rather_than_assuming_it(
    notebook: dict[str, Any],
) -> None:
    """The bug this replaces: chdir into a hard-coded folder raised a bare
    FileNotFoundError whenever only the notebook itself had been uploaded."""
    cell = find_cell(notebook, "def find_project")
    assert 'Path("/content/' not in cell, "the project path is hard-coded again"
    assert 'utils.py").is_file()' in cell, "a candidate must be checked before it is entered"


def test_the_bootstrap_stops_when_the_project_is_not_there(
    notebook: dict[str, Any],
) -> None:
    """The bug this replaces: it printed the upload steps and carried on, so the
    run died one cell later on `import src`. That reads as a different problem,
    and the message explaining it has already scrolled away."""
    cell = find_cell(notebook, "def find_project")
    assert "raise RuntimeError" in cell, "a printed warning alone gets walked past"


@pytest.mark.parametrize(
    "call",
    ["evaluate_run(RUN_NAME, CSV_PATH)", "train_with_lightning(final, CSV_PATH"],
)
def test_every_cell_that_rereads_the_csv_is_given_the_platform_path(
    notebook: dict[str, Any], call: str
) -> None:
    """The bug this replaces: these two rebuild the split from train.csv rather
    than reusing `splits`, so without a path they read data/raw/train.csv. On
    Kaggle the file is under /kaggle/input, so sections 6 and 7 failed there
    while every other cell worked."""
    assert any(call in source for source in sources(notebook, "code")), (
        f"{call} must be passed the CSV the bootstrap found"
    )


def test_no_training_cell_hard_codes_its_epoch_budget(notebook: dict[str, Any]) -> None:
    """The bug this replaces: five cells each carried their own epoch literal, so
    shortening the notebook for a trial run meant finding all five, and missing
    one left a cell training for fifty epochs in the middle of a quick pass."""
    literals = [
        line.strip()
        for source in sources(notebook, "code")
        for line in source.splitlines()
        if re.search(r"max_epochs=\d", line)
    ]
    assert not literals, f"these cells set their own epoch budget: {literals}"


def test_the_gpu_section_points_at_the_readme(notebook: dict[str, Any]) -> None:
    """The notebook defers the cloud detail to the README, which must cover it."""
    assert "## Running on a cloud GPU" in (PROJECT_ROOT / "README.md").read_text()
    assert any("cloud GPU" in source for source in sources(notebook, "markdown"))


def test_the_readme_epoch_table_matches_the_notebook(notebook: dict[str, Any]) -> None:
    """The README prints the notebook's epoch budgets, which it cannot read.

    The bug this replaces: the budgets were raised in the notebook and the
    README went on quoting the old ones, so anyone planning a GPU session
    budgeted for half the run.
    """
    setup = "\n".join(sources(notebook, "code"))
    budgets = re.findall(r"(\w*EPOCHS) = (\d+) if QUICK else (\d+)", setup)
    assert budgets, "no `X = quick if QUICK else full` epoch budget found in the notebook"

    readme = (PROJECT_ROOT / "README.md").read_text()
    rows = re.findall(r"^\|([^|]+)\|([^|]+)\|([^|]+)\|$", readme, flags=re.MULTILINE)
    for name, quick, full in budgets:
        assert any(
            quick in cell_quick and full in cell_full for _, cell_quick, cell_full in rows
        ), f"the README has no table row quoting {name} = {quick} quick / {full} full"


def test_every_notebook_file_reference_resolves() -> None:
    """Markdown that names a project file must name one that exists."""
    text = NOTEBOOK.read_text()
    for reference in set(re.findall(r"`(src/[a-z_]+\.py)`", text)):
        assert (PROJECT_ROOT / reference).exists(), f"{reference} is referenced but missing"


def test_the_improvements_table_covers_what_was_built(notebook: dict[str, Any]) -> None:
    """Section 9 claims these are implemented; the modules must be present."""
    closing = find_cell(notebook, "## 9. Limitations and further work")
    for module in ("src/augment.py", "src/ensemble.py", "src/reference_cnn.py"):
        assert Path(PROJECT_ROOT / module).exists()
    for claim in ("Learning-rate schedule", "Data augmentation", "Seed ensemble", "convolutional"):
        assert claim in closing, f"section 9 no longer mentions {claim}"


def test_every_name_a_cell_uses_is_defined_by_an_earlier_cell(
    notebook: dict[str, Any],
) -> None:
    """The cells are a narrative, so they must run top to bottom in order.

    A cell that reads a name defined only in a later cell works while you are
    editing — the name is still in the kernel — and fails for everyone who runs
    the notebook fresh. That is the classic way a notebook rots.
    """
    import builtins

    def bound_names(tree: ast.AST) -> set[str]:
        """Names a cell binds: assignments, imports, defs and loop targets."""
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        names |= {
            alias.asname or alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import | ast.ImportFrom)
            for alias in node.names
        }
        names |= {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        }
        # `except X as error` binds a name too. Python drops it again at the end
        # of the handler, but this test is about the order the cells run in, and
        # a cell that binds it is not a cell that depends on an earlier one.
        return names | {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.name
        }

    defined = set(dir(builtins)) | {"__name__", "_", "get_ipython"}

    for index, source in enumerate(sources(notebook, "code")):
        tree = ast.parse(strip_magics(source))
        used = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        # A cell may use what it binds itself; only cross-cell order is at stake.
        undefined = used - defined - bound_names(tree)
        assert not undefined, f"code cell {index} uses {sorted(undefined)} before they exist"

        defined |= bound_names(tree)


def test_the_notebook_asks_for_the_inline_backend(notebook: dict[str, Any]) -> None:
    """The bug this replaces: no cell named a backend, so the figures went
    wherever the session's default sent them -- and `src/evaluate.py` used to
    force the file-only Agg backend at import time, which the setup cell
    imports. Every plot in the report then rendered into a buffer nobody sees.

    Saying `%matplotlib inline` makes the notebook independent of that, and it
    has to be said before the first figure is drawn to be worth anything.
    """
    code = sources(notebook, "code")
    chooses = [index for index, source in enumerate(code) if "%matplotlib inline" in source]
    plots = [index for index, source in enumerate(code) if "plt." in source]

    assert chooses, "no cell selects a backend; the figures go wherever the session decides"
    assert plots, "no cell draws a figure any more"
    assert chooses[0] < min(plots), "the backend is chosen after the first figure is drawn"

    # Within that cell it has to come last. A module calling matplotlib.use() at
    # import time overrides whatever was set before it, so a magic above the
    # imports is only as good as every module they pull in.
    setup = code[chooses[0]]
    assert setup.index("%matplotlib inline") > setup.rindex("\nfrom src."), (
        "the backend is set above the src imports, where any of them can override it"
    )


def test_the_unpack_cell_runs_before_the_bootstrap(notebook: dict[str, Any]) -> None:
    """The bug this replaces: the unzip was pasted into a cell below the
    bootstrap, so Run all reached `find_project` while the code was still inside
    the archive. It raised "is not on this machine" and the unzip never ran.

    The cell also has to recognise its own archive: a session usually holds
    other zips, and the competition data is one of them.
    """
    code = sources(notebook, "code")
    unpack = next(index for index, source in enumerate(code) if "extractall" in source)
    bootstrap = next(index for index, source in enumerate(code) if "def find_project" in source)

    assert unpack < bootstrap, "the bundle is unpacked after the cell that looks for it"
    assert "src/utils.py" in code[unpack], "any zip in the session would be extracted"


def test_every_figure_is_emitted_explicitly(notebook: dict[str, Any]) -> None:
    """A cell ending in `figure.tight_layout()` returns None, and the figure only
    appears because the inline backend sweeps up open figures after the cell.
    That is a property of one frontend, not of the notebook: under nbconvert or
    a plain backend the same cell renders nothing and the report is a wall of
    text. `plt.show()` says it outright.
    """
    for index, source in enumerate(sources(notebook, "code")):
        if "plt.subplots(" not in source:
            continue
        assert source.rstrip().endswith("plt.show()"), (
            f"code cell {index} draws a figure but never shows it"
        )


def test_the_committed_notebook_was_run_start_to_finish(notebook: dict[str, Any]) -> None:
    """A half-run notebook is exactly what a reader opens, and it reads as unfinished.

    Execution counts record the order the cells were run in. `make notebook-run`
    leaves them 1, 2, 3 and so on with none missing, so a None means a cell was
    never run and a count out of order means the notebook was driven by hand.
    Either way the outputs below that point belong to some other state of the
    code.
    """
    counts = [
        cell["execution_count"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "".join(cell["source"]).strip()
    ]
    never_run = counts.count(None)
    assert not never_run, (
        f"{never_run} code cells were never run. Execute the whole notebook with "
        "`make notebook-run` before committing it."
    )
    assert all(earlier < later for earlier, later in zip(counts, counts[1:], strict=False)), (
        f"cells were run out of order (counts {counts}). Re-run with `make notebook-run`."
    )
