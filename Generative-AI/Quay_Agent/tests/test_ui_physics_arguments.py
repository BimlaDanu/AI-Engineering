r"""A page that has a coupling must hand it to whatever grades or exports with it.

Every routine that computes a reference energy or writes a circuit out takes the
Hamiltonian strengths as *optional* arguments, defaulted to the calibration point
:math:`J = h = 1`, :math:`g = 0`. That default is right for a caller that genuinely
has no couplings, and it is silent, so a caller that has them and forgets to pass
them gets an answer to a neighbouring problem with no error anywhere.

Two pages had forgotten, and neither failure looked like a missing argument:

* The Lab page ran its solves with the longitudinal field :math:`g` from the settings
  knob and computed the exact answer without it. There is no exact route at
  :math:`g \neq 0`, so the number that came back was the :math:`g = 0` energy, which
  is higher. A correct run therefore landed *below* it and the page printed its own
  "the result is below the exact answer" warning -- the signature it uses for a bug --
  about a run that had none.
* The Machines page priced a circuit at the knob's :math:`J` and :math:`h` and then
  exported OpenQASM at the default ones, under a caption promising that everything
  the estimate charged for is in the file. The rotation angles are :math:`2\gamma J`
  and :math:`2\beta h`, so the exported file was a circuit for a different problem.

Both are the defect of §3n in a smaller place: the shape of the problem travelling
separately from the problem. Rendering a page to catch it costs a minute or more, and
what went wrong is visible in the call itself, so these read the source instead. The
check is that the argument is *supplied*, which is the part a person forgets; whether
the value is right is the business of the tests for the functions themselves.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.hardware.export import qasm3, run_card
from src.ui.panels import true_energy
from src.ui.status import PROJECT_ROOT

PAGES = PROJECT_ROOT / "src" / "ui" / "pages"

REQUIRED_ARGUMENTS = {
    "true_energy": ("longitudinal_field",),
    "qasm3": ("couplings",),
    "run_card": ("couplings",),
}
"""Which keyword each call has to carry, by the name being called.

Keyed on the trailing name so that ``panels.true_energy(...)`` and a bare
``true_energy(...)`` are the same entry. Only the arguments that carry *physics*
are listed: a missing ``note`` is a type error and needs no test, whereas a missing
coupling is a plausible number computed for the wrong Hamiltonian.
"""

CALLEES: dict[str, Callable[..., Any]] = {
    "true_energy": true_energy,
    "qasm3": qasm3,
    "run_card": run_card,
}
"""The functions themselves, so the parameter order comes from the signature.

Read rather than restated, because a positional argument is only identifiable by
position and a list copied here would go stale the first time one is inserted.
"""


def _called_name(call: ast.Call) -> str:
    """Get the trailing name of whatever a call names, dotted path or not.

    Args:
        call: The call node.

    Returns:
        ``"true_energy"`` for both ``true_energy(...)`` and ``panels.true_energy(...)``,
        and the empty string for anything not called by name -- a call on a subscript
        or on another call, neither of which appears in these pages.
    """
    target = call.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _positional_names(call: ast.Call, parameters: list[str]) -> set[str]:
    """Work out which parameters a call filled positionally.

    Args:
        call: The call node.
        parameters: The callee's parameter names, in order.

    Returns:
        The names covered by the call's positional arguments.
    """
    return set(parameters[: len(call.args)])


def _parameters_of(name: str) -> list[str]:
    """Look up a callee's parameter order so positional arguments can be named.

    Args:
        name: The trailing name of the function being called.

    Returns:
        The parameter names in declaration order.
    """
    target: Callable[..., Any] = CALLEES[name]
    # `true_energy` is wrapped by Streamlit's cache decorator, which reports the
    # wrapper's `*args, **kwargs` rather than the parameters the page actually
    # writes. `__wrapped__` is the undecorated function underneath.
    return list(inspect.signature(getattr(target, "__wrapped__", target)).parameters)


def _calls_in(path: Path) -> list[tuple[str, ast.Call]]:
    """Find every call in one page that this test has an opinion about.

    Args:
        path: The page's source file.

    Returns:
        Pairs of called name and call node, for the names in
        :data:`REQUIRED_ARGUMENTS` only.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _called_name(node)
            if name in REQUIRED_ARGUMENTS:
                found.append((name, node))
    return found


@pytest.mark.parametrize("page", sorted(PAGES.glob("*.py")), ids=lambda path: path.stem)
def test_every_page_hands_over_the_couplings_it_holds(page: Path) -> None:
    """No page may grade or export against a coupling it did not supply.

    Args:
        page: One page module.
    """
    for name, call in _calls_in(page):
        parameters = _parameters_of(name)
        supplied = _positional_names(call, parameters) | {
            keyword.arg for keyword in call.keywords if keyword.arg
        }
        for required in REQUIRED_ARGUMENTS[name]:
            assert required in supplied, (
                f"{page.name}:{call.lineno} calls {name}() without {required!r}, so it "
                f"falls back to the calibration point and answers about a different "
                f"Hamiltonian than the one this page is set to"
            )


def test_the_guard_is_actually_looking_at_something() -> None:
    """The pages really do hold calls of each kind this test guards.

    The check above walks whatever it finds, so it would pass on a set of pages
    holding none of these calls at all -- a rename, a move, or a typo in
    :data:`REQUIRED_ARGUMENTS` would turn it green rather than red. This names the
    calls that exist today so that losing one is a failure.
    """
    seen = {name for page in PAGES.glob("*.py") for name, _ in _calls_in(page)}
    assert seen == set(REQUIRED_ARGUMENTS), (
        f"expected every guarded call to appear in some page; found {sorted(seen)}"
    )


def test_the_guard_would_notice_an_omission() -> None:
    """The check fails on a call that leaves the coupling out.

    A structural test that never fires is indistinguishable from one that cannot,
    so this asserts the omission is actually detected rather than trusting that it
    would be.
    """
    call = ast.parse("panels.true_energy(n, J, h, boundary)").body[0]
    assert isinstance(call, ast.Expr)
    assert isinstance(call.value, ast.Call)
    parameters = _parameters_of("true_energy")
    supplied = _positional_names(call.value, parameters)
    assert "longitudinal_field" not in supplied
