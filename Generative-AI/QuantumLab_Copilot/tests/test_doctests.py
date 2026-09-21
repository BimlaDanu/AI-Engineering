r"""Every ``>>>`` example in ``src/`` is executed here.

The docstrings in this project carry worked examples -- ``heuristic_route`` shows the
route it picks, ``shelf_names`` shows the registry, ``parse_body`` shows both wire
formats -- and an example is a claim about behaviour. Nothing ran them. There were
195 of them across 28 modules, and three had been wrong for as long as they had
existed:

* ``physics/ed.py`` compared ``levels[1] - levels[0] > 0.0`` against ``True``, which
  NumPy 2 prints as ``np.True_``. The physics was right and the transcript was stale.
* ``tools/http.py`` and ``tools/websearch.py`` had doubled their escapes inside
  *raw* docstrings, so ``\\n`` reached the example as a literal backslash and an
  ``n``. ``parse_body`` was being shown a one-line event stream with no data in it
  and raising, and ``clean_query`` was being shown no tab to collapse. Both examples
  demonstrated the opposite of what they claimed.

None of that is catchable by review -- a doubled backslash in a docstring is
invisible -- and all of it is caught by running the examples. So they run.

Parametrised by module so a failure names the module rather than reporting "a
doctest somewhere failed", and the runner's own report is handed to the assertion,
because the useful part of a doctest failure is the expected-versus-got block.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil
from io import StringIO

import pytest

import src

OPTIONS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
"""How the examples are read.

``ELLIPSIS`` so an example may write ``...`` for a part that is not the point, and
``NORMALIZE_WHITESPACE`` so re-wrapping a long expected value to fit the line limit
is a formatting change rather than a test failure. Neither weakens a claim: they
describe how the *transcript* is compared, not what is run.
"""


SCRIPTS = ("src.ui.app", "src.ui.pages.")
"""Module paths that are Streamlit *scripts* rather than modules.

Importing one of these runs the application. That is what a Streamlit script is:
top-level statements that draw a page, with nothing exported -- so there is no
example in any of them to check, and importing them was only ever a side effect of
sweeping the package.

It was a harmless side effect until it was not. The entry point draws the account
controls, one of which holds a key form, and ``st.form`` entered without a script
run context leaves form state on Streamlit's process-wide singletons: every
``AppTest`` that ran afterwards in the same process then raised ``st.button() can't
be used in an st.form()`` from somewhere entirely unrelated -- sixty-one failures,
in two other files, none of them near the cause.

So the scripts are excluded here, and the thing that exercises them is
``AppTest``, which is what they are written for. :func:`src.ui.panels` and
:mod:`src.ui.identity` are ordinary modules and stay in the sweep.
"""


def module_names() -> list[str]:
    """Every importable module under ``src``, minus the Streamlit scripts.

    Returns:
        Dotted names, sorted, packages excluded -- a package's ``__init__`` carries
        no examples here and importing it happens anyway -- and :data:`SCRIPTS`
        excluded, because importing one of those runs the application.
    """
    return sorted(
        info.name
        for info in pkgutil.walk_packages(src.__path__, prefix="src.")
        if not info.ispkg and not info.name.startswith(SCRIPTS)
    )


def test_the_module_list_is_not_empty() -> None:
    # The failure this guards: a change to the package layout makes walk_packages
    # return nothing, every parametrised case disappears, and a suite that now
    # checks no examples at all still reports green.
    assert len(module_names()) > 40


@pytest.mark.parametrize("name", module_names())
def test_the_examples_in_the_docstrings_are_true(name: str) -> None:
    """Run one module's examples and fail with the runner's own report."""
    module = importlib.import_module(name)
    report = StringIO()
    runner = doctest.DocTestRunner(optionflags=OPTIONS)
    for test in doctest.DocTestFinder().find(module):
        runner.run(test, out=report.write)
    assert runner.failures == 0, f"{name}:\n{report.getvalue()}"
