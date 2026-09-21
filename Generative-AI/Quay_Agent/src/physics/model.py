"""The problem specification every physics method consumes.

A single frozen dataclass is the contract between the agent layer and the
physics layer. The agent's job is to turn a sentence into one of these; the
physics layer's job is to turn one of these into a number. Keeping the two
sides apart behind an immutable, validated value object is what makes the
result cacheable, the run reproducible and the whole thing testable without a
language model in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from src.physics.lattice import MIN_SIDE, Geometry, Lattice

BoundaryCondition = Literal["periodic", "open"]
"""Lattice boundary condition. Periodic closes the chain into a ring."""


HAMILTONIAN_LATEX = (
    r"\hat H = -J \sum_{\langle ij \rangle} \hat\sigma^z_i \hat\sigma^z_j "
    r"- h \sum_i \hat\sigma^x_i"
)
r"""The project's Hamiltonian as LaTeX, with no delimiters around it.

Two terms, and that is the whole model as a reader meets it. A coupling $J$ that
makes neighbouring spins agree, and a transverse field $h$ that knocks them over.
The first sum runs over $\langle ij \rangle$, meaning *each pair of neighbouring
sites once*, so the same equation covers every shape the project supports: a line,
a square lattice, a triangular lattice. One string, imported by the page caption,
the agent's self-description and the answer it composes, because the same equation
written out three times drifts three ways -- and the version a reader met first is
the one they will trust.

Operators are written :math:`\hat\sigma^z_i`, not ``Z_i``: both are standard and
mean the same Pauli matrix, but the hatted form is what a physics reader expects to
see in a Hamiltonian, and the letter form reads as a *gate* to anyone coming from
the quantum-computing side.

``J`` is kept even though it is 1 by default: it is a setting the user can change,
and an equation that hides the coupling cannot explain the number that comes back
when they do.

The one-dimensional case is written out separately in
:data:`HAMILTONIAN_CHAIN_LATEX`, and the longitudinal term in
:data:`HAMILTONIAN_WITH_LONGITUDINAL_LATEX`. Both are variants of this equation
rather than different models.
"""

HAMILTONIAN_CHAIN_LATEX = (
    r"\hat H = -J \sum_{i=1}^{L} \hat\sigma^z_i \hat\sigma^z_{i+1} "
    r"- h \sum_{i=1}^{L} \hat\sigma^x_i"
)
r"""The same Hamiltonian on a line, where the neighbour sum can be written out.

In one dimension each site has one forward neighbour, so $\langle ij \rangle$
becomes $i, i+1$ and the sum carries explicit limits. This is a *limiting case* of
:data:`HAMILTONIAN_LATEX`, not a second model, and it is written out separately for
two reasons.

The first is that it is the only shape with a closed-form solution. The
Jordan-Wigner transformation turns a line of interacting spins into free fermions
and turns nothing else into free fermions, so
:mod:`src.physics.reference.free_fermions` applies here and nowhere else. That gives
the project a reference answer to compare against at any length, which is what makes
its accuracy claims checkable rather than asserted.

The second is that a good deal of well-understood physics lives in this case and can
be worked through by hand: the gap closing linearly at $h = J$, the exact critical
point pinned by Kramers-Wannier duality, the logarithmic divergence of the
susceptibility, the mapping onto a two-dimensional classical Ising model. A reader
who follows the one-dimensional case has the vocabulary for the two-dimensional ones,
where no closed form exists.
"""

HAMILTONIAN_WITH_LONGITUDINAL_LATEX = (
    r"\hat H = -J \sum_{\langle ij \rangle} \hat\sigma^z_i \hat\sigma^z_j "
    r"- g \sum_i \hat\sigma^z_i "
    r"- h \sum_i \hat\sigma^x_i"
)
r"""The same Hamiltonian with the longitudinal field, for when it is switched on.

Rendered in place of :data:`HAMILTONIAN_LATEX` wherever ``g`` is non-zero, and kept
in the corpus regardless -- a question about what the longitudinal field does gets an
answer whether or not the control in front of the reader has moved.

The term is not in the default equation, and that is a decision rather than an
omission. On a page, the equation tells a reader what problem is being solved, and by
default this project solves one with a single dimensionless knob, $h/J$. A third
symbol that is zero everywhere, contributes nothing to any number on any screen, and
can only be changed from an advanced control reads as a third free parameter, which
makes the problem look harder than it is.

What the term is for is still served by keeping it reachable. On a line at $g = 0$
the model is integrable, so an honest feasibility verdict about it is "no quantum
advantage"; $g \ne 0$ removes that integrability at no cost in two-qubit depth. The
solvers keep their ``longitudinal_field`` arguments, and the surfaces render this
form the moment ``g`` is non-zero.

``g`` is not carried by :class:`TFIMSpec`. It is passed at the point of use --
:func:`src.physics.quantum.hamiltonians.ising_chain` and the classical baseline both
take a ``longitudinal_field`` argument -- because the two reference solvers in
:mod:`src.physics.reference` do not implement it, and a spec able to express a
problem that nothing in the project can check is a spec that invites an uncheckable
run.
"""

HAMILTONIAN_CHAIN_WITH_LONGITUDINAL_LATEX = (
    r"\hat H = -J \sum_{i=1}^{L} \hat\sigma^z_i \hat\sigma^z_{i+1} "
    r"- g \sum_{i=1}^{L} \hat\sigma^z_i "
    r"- h \sum_{i=1}^{L} \hat\sigma^x_i"
)
r"""The one-dimensional case with the longitudinal field.

The fourth corner of a two-by-two: general or one-dimensional, with the
longitudinal term or without. It exists so that a page describing a line with
``g`` switched on shows the sum it actually evaluates, rather than a neighbour-pair
notation that hides the one thing a line makes explicit.
"""

HAMILTONIAN_INLINE = f"${HAMILTONIAN_LATEX}$"
"""The Hamiltonian for the middle of a sentence."""


def hamiltonian_latex(
    longitudinal_field: float = 0.0,
    geometry: Geometry | None = None,
) -> str:
    r"""The Hamiltonian to render, given the shape and whether ``g`` is on.

    One function rather than a conditional at each call site. Several surfaces show
    this equation -- the chat page, the lab page, the report and the composed answer
    -- and four copies of the same two-way branch is three copies that will
    eventually disagree about which problem the reader is looking at.

    Args:
        longitudinal_field: The value of ``g`` in the run being described. Zero,
            which is the default everywhere in the application, leaves the third
            term out.
        geometry: The shape being solved, when one is known. ``"chain"`` writes the
            neighbour sum out as ``i, i+1`` with explicit limits, which is what a
            one-dimensional problem should show. ``None`` -- the default -- gives
            the neighbour-pair form that is true of every shape, and is what to use
            when describing the model rather than a particular run.

    Returns:
        The LaTeX, with no delimiters.

    Examples:
        The probe is the longitudinal *term*, not the letter ``g``: the two-term
        form contains a ``g`` in every ``\sigma`` it prints, so ``"g" in ...`` is
        true of both forms and tests nothing.

        >>> r"- g \sum" in hamiltonian_latex()
        False
        >>> r"- g \sum" in hamiltonian_latex(0.4)
        True

        A shape decides how the neighbour sum is written:

        >>> r"\langle ij \rangle" in hamiltonian_latex()
        True
        >>> r"\hat\sigma^z_{i+1}" in hamiltonian_latex(geometry="chain")
        True
        >>> r"\langle ij \rangle" in hamiltonian_latex(geometry="square")
        True
    """
    on_a_line = geometry == "chain"
    if longitudinal_field == 0.0:
        return HAMILTONIAN_CHAIN_LATEX if on_a_line else HAMILTONIAN_LATEX
    if on_a_line:
        return HAMILTONIAN_CHAIN_WITH_LONGITUDINAL_LATEX
    return HAMILTONIAN_WITH_LONGITUDINAL_LATEX


def hamiltonian_display(
    longitudinal_field: float = 0.0,
    geometry: Geometry | None = None,
) -> str:
    """The Hamiltonian as its own centred line, given the shape and the field.

    Args:
        longitudinal_field: The value of ``g`` in the run being described.
        geometry: The shape being solved, when one is known. See
            :func:`hamiltonian_latex`.

    Returns:
        The LaTeX wrapped in ``$$`` for Streamlit's display maths. The delimiters
        live here rather than at the call site for the reason given on
        :data:`HAMILTONIAN_DISPLAY`.
    """
    return f"$$\n{hamiltonian_latex(longitudinal_field, geometry)}\n$$"


def fields_in_words(
    coupling: float,
    transverse_field: float,
    longitudinal_field: float = 0.0,
    *,
    maths: bool = False,
) -> str:
    """Name the couplings of one chain, leaving out the field that is not there.

    Every composed answer states which chain it is about, and for most of this
    project's runs that is two numbers rather than three: :math:`g` defaults to
    zero and the pages no longer show it. Printing ``g = 0`` anyway invites the
    reader to ask what :math:`g` is, gets no answer -- the term is not in the
    Hamiltonian they were shown -- and then invites a follow-up about a knob the
    question never mentioned. Reporting a term as present at zero is a different
    claim from not having it, and the second one is what is true here.

    Written as a function taking three floats rather than a method, because the
    three callers hold three different objects -- a formal model, a method race
    and a report -- and none of them should have to grow a common base class to
    agree on how a chain is named.

    Args:
        coupling: :math:`J`, the neighbour interaction.
        transverse_field: :math:`h`, the field across the coupling direction.
        longitudinal_field: :math:`g`, the field along it. Omitted when zero,
            which is the default and the case the closed-form grader covers.
        maths: Wrap each symbol in ``$`` so Streamlit renders it as italic
            mathematics. False for a plain-text note or a prompt.

    Returns:
        A comma-separated phrase, of the form ``J = 1, h = 1`` or
        ``J = 1, h = 1, g = 0.4``.
    """

    def term(symbol: str, value: float) -> str:
        text = f"{symbol} = {value:g}"
        return f"${text}$" if maths else text

    parts = [term("J", coupling), term("h", transverse_field)]
    if longitudinal_field != 0.0:
        parts.append(term("g", longitudinal_field))
    return ", ".join(parts)


HAMILTONIAN_DISPLAY = f"$$\n{HAMILTONIAN_LATEX}\n$$"
"""The Hamiltonian as its own centred line.

Streamlit renders ``$$`` as display math and ``$`` as inline, and nothing else:
``\\(...\\)`` and ``\\[...\\]`` are shown as literal backslashes and brackets. The
delimiters therefore live here rather than at each call site, where the wrong pair
is a rendering bug nobody notices until a screenshot.
"""

DEFAULT_SITES = 6
"""Chain length assumed when a caller does not name one.

Deliberately small, and chosen for test runs rather than for physics. Every method in
the project is cheap here -- sparse diagonalisation faces a 64 x 64 matrix, the
free-fermion solver does six terms of arithmetic, and the Monte Carlo sampler works a
six-column lattice -- so a suite that exercises all three costs seconds rather than
minutes, and a doctest can afford to actually run a solver.

Six rather than four because it is the smallest even length at which the chain is not
degenerate in the ways that hide bugs: it has interior sites as well as ends, it is
even so a periodic ring stays bipartite and the checkerboard sweep is legal, and it is
long enough that an open chain and a ring give visibly different energies. Anything
smaller passes tests that a real chain would fail.

It is a *default*, not a cap -- see :data:`MAX_SITES_STATEVECTOR` for that -- and any
result being reported rather than tested should name its size explicitly.
"""

MAX_SITES_STATEVECTOR = 16
"""Hard cap on ``L`` for any method that simulates a circuit on ``2**L`` amplitudes.

**This is a time limit, not a memory limit**, which is why it sits where it does.
A state vector of sixteen qubits is 1 MB and could not trouble any machine; what
grows is the work of applying a layer of gates to it. Measured on this machine, one
energy-and-gradient evaluation of the ansatz costs

===== ======================= ================== ========================
``L``  energy + gradient       60 optimiser steps  the three-method race
===== ======================= ================== ========================
12     3.0 ms                  0.2 s               0.5 s
14     9.4 ms                  0.6 s               1.7 s
**16** **39 ms**               **2.3 s**           **7.0 s**
18     308 ms                  18 s                55 s
===== ======================= ================== ========================

Sixteen is where that table stops being usable in a browser session. Eighteen puts
a single question at nearly a minute of arithmetic, which is the latency this
project has already had to hunt down once; sixteen keeps the whole three-method
comparison inside ten seconds.

Raised from twelve in Phase 2 of the 2D work, and the reason is geometry rather
than generosity: a ``4 x 4`` square lattice is exactly sixteen sites, and it is the
smallest two-dimensional cluster with an interior -- every site of a ``3 x 3`` is on
its boundary. A cap of twelve would have refused the one 2D size worth asking about.

That is the *cap*, not the size to write things at. :data:`WORKING_SITES` is twelve
and everything routine belongs there; sixteen is for the cases where the size is
itself the point.

**A real device would not have this limit at all**, which is the entire feasibility
question. The refusal is a property of the *simulation*, and
:func:`src.physics.method_catalogue.variational_quantum_eigensolver_unsupported_reason`
says so in those words so that the distinction stays visible to a reader.
"""

MAX_SITES_SPARSE = 16
"""Hard cap on ``L`` for sparse exact diagonalisation -- the Lanczos route.

The project's working size is twelve. This is the exception to it, and it is the
only one: a chain above twelve sites is answered by Lanczos or it is not answered
at all. See :data:`WORKING_SITES` for why the line sits there.

Equal to :data:`MAX_SITES_STATEVECTOR` rather than above it, which needs saying
because the two methods are limited by different things and used to differ. Exact
diagonalisation never applies a gate: it assembles a matrix with at most ``L + 1``
non-zeros per row and hands it to a Lanczos iteration, so its cost is dominated by
the sparse matrix-vector product and it *could* go further. Measured on this
machine, for the ground state alone:

===== ============== ============ ============= ==================
``L``  non-zeros      CSR storage  build + solve peak process memory
===== ============== ============ ============= ==================
12     53,248         0.4 MB       0.01 s        --
**16** **1,114,112**  **8.9 MB**   **0.26 s**    **155 MB**
18     4,980,736      40 MB        2.5 s         532 MB
20     22,020,096     176 MB       12 s          1.9 GB
24     --             --           215 s         8.6 GB
===== ============== ============ ============= ==================

Sixteen is the last row that is free. Eighteen puts a single question at two and a
half seconds and half a gigabyte; twenty costs twelve seconds and 1.9 GB, which is
a test suite that flakes under load rather than a method that works. Both are
refused rather than attempted: an agent that declines a run it cannot afford is
more useful than one that hangs.

Sixteen is also exactly a ``4 x 4`` square, the smallest two-dimensional cluster
with an interior -- every site of a ``3 x 3`` is on its boundary. Verified at
0.19 s. A ``4 x 5`` at twenty sites is what this cap now refuses, and refusing it
is the point: nothing in this project needs a size whose cost is measured in
gigabytes.

The cap is enforced by each method's applicability check rather than here, because
methods whose cost is ``O(L)`` -- the free-fermion solver -- are legitimately
unbounded and must not be caught by it.

**The third limit in this project is not a site count**, and it is worth saying so
here because a reader looking for one will not find it. Whether a matrix is
diagonalised densely or iteratively is decided on the Hilbert-space *dimension* by
:data:`~src.physics.reference.exact_diagonalisation.DENSE_SOLVER_MAX_DIMENSION`,
which is the better formulation: a dense eigensolve on a 64 x 64 matrix is instant
and unconditionally reliable, and below that size a Lanczos iteration is both
pointless and prone to convergence warnings. No method here ever materialises a
dense ``2**L x 2**L`` matrix above that threshold, so there is no third site cap to
set.
"""

WORKING_SITES = 12
"""The size everything is expected to run at, and the ceiling on every test.

Not a cap -- no method checks against it -- but the number that decides what gets
written. Two caps above it are real and enforced
(:data:`MAX_SITES_STATEVECTOR`, :data:`MAX_SITES_SPARSE`, both sixteen); this is
the one that decides what anybody *asks for*, and the rule attached to it is
short:

* Twelve sites for every test, every default, every example and every eval case.
* Sixteen only where the size is the point -- a ``4 x 4`` square, or a check that
  a cap is enforced at its own boundary.
* Above sixteen, nothing. Twenty sites was offered until it cost a UI test three
  minutes and a flake, which is what a size chosen for looking impressive buys.

Twelve rather than ten or fourteen because it is the largest length whose whole
bench is instant: exact diagonalisation in 0.01 s, one energy-and-gradient in
3.0 ms, the three-method race in half a second. Every one of those grows by
roughly four per two sites added, so twelve is the last size at which a suite can
run the *real* methods rather than mocks of them.
"""


@dataclass(frozen=True, slots=True)
class TFIMSpec:
    r"""A fully specified transverse-field Ising model on a named lattice.

    The Hamiltonian, in the convention used everywhere in this project, is

    .. math::

        H = -J \sum_{\langle ij \rangle} \sigma^z_i \sigma^z_j
          \; - \; h \sum_i \sigma^x_i \; - \; g \sum_i \sigma^z_i

    where :math:`\sigma^z` and :math:`\sigma^x` are Pauli matrices and
    :math:`\langle ij \rangle` runs over each neighbouring pair once. Published
    work is split between this convention and the mirrored one, which puts the
    coupling on :math:`\sigma^x` and the field on :math:`\sigma^z`. The two are
    related by a global basis rotation and give identical spectra, but mixing them
    silently is a classic source of wrong answers. This project commits to the form
    above and converts at the boundary rather than tolerating both internally.

    The instance is frozen so it can be hashed into a run cache: two identical
    specs must always produce the same answer.

    The shape is a field here rather than an argument passed alongside. As an
    argument, every method took a spec and, separately, an optional geometry
    defaulting to a line, so a caller who had a shape and forgot to pass it got a
    silently different problem of the same size. A sixteen-site problem asked about
    a ``4 x 4`` square was told the closed-form solution applied to it, because the
    predicate that refuses the closed form had only a spec to look at and every spec
    was a line. The
    closed form exists for a line and for nothing else, and its answer at sixteen
    sites is :math:`-20.40` where the square's is :math:`-34.01`. A genuine
    variational energy for the square therefore came back *below* the number
    reported as exact, which reads as a broken variational bound and is really a
    comparison between two different problems. Putting the shape in the spec is
    what makes that mistake unrepresentable: there is one problem statement, and
    everything that describes, prices, solves or grades it reads the shape off the
    same object.

    Attributes:
        n_sites: Total number of spins, ``L``. At least 2. For a two-dimensional
            lattice this is ``rows * cols``, so it stays the single authority on
            how big the problem is and no consumer needs to multiply anything.
        coupling: The Ising coupling ``J``. Strictly positive (ferromagnetic).
        field: The transverse field ``h``. Non-negative.
        boundary: ``"periodic"`` (edges wrap) or ``"open"`` (edges are edges).
        geometry: Which shape the sites sit on -- ``"chain"``, ``"square"`` or
            ``"triangular"``. Defaults to a chain, which is what every existing
            caller means and is why this field could be added without touching
            them.
        rows: Rows of sites for a two-dimensional lattice, and one for a chain.
            The columns follow from ``n_sites``, so the two cannot disagree about
            the size. Stored rather than inferred because sixteen sites is a chain,
            a ``4 x 4`` square and a ``2 x 8`` rectangle, and guessing between them
            would put a shape nobody asked for into a report.
    """

    n_sites: int = DEFAULT_SITES
    coupling: float = 1.0
    field: float = 1.0
    boundary: BoundaryCondition = "periodic"
    geometry: Geometry = "chain"
    rows: int = 1

    def __post_init__(self) -> None:
        """Reject physically or numerically meaningless specifications.

        Raises:
            TypeError: If ``n_sites`` or ``rows`` is not an integer.
            ValueError: If any field is outside its permitted range, if the rows
                do not divide the sites into a rectangle -- which would describe a
                ragged lattice that no solver here can build bonds for -- or if the
                columns they leave are too few for the shape to be two-dimensional.
        """
        if not isinstance(self.n_sites, int) or isinstance(self.n_sites, bool):
            raise TypeError(f"n_sites must be an int, got {type(self.n_sites).__name__}")
        if self.n_sites < 2:
            raise ValueError(f"n_sites must be at least 2, got {self.n_sites}")
        if self.coupling <= 0.0:
            raise ValueError(f"coupling J must be strictly positive, got {self.coupling}")
        if self.field < 0.0:
            raise ValueError(f"field h must be non-negative, got {self.field}")
        if self.boundary not in get_args(BoundaryCondition):
            raise ValueError(
                f"boundary must be one of {get_args(BoundaryCondition)}, got {self.boundary!r}"
            )
        if self.geometry not in get_args(Geometry):
            raise ValueError(f"geometry must be one of {get_args(Geometry)}, got {self.geometry!r}")
        if not isinstance(self.rows, int) or isinstance(self.rows, bool):
            raise TypeError(f"rows must be an int, got {type(self.rows).__name__}")
        if self.rows < 1:
            raise ValueError(f"rows must be at least 1, got {self.rows}")
        if self.geometry == "chain" and self.rows != 1:
            raise ValueError(f"a chain has one row, got {self.rows}")
        if self.n_sites % self.rows != 0:
            raise ValueError(
                f"{self.rows} rows do not divide {self.n_sites} sites into a rectangle"
            )
        if self.geometry != "chain" and self.rows < 2:
            raise ValueError(
                f"a {self.geometry} lattice needs at least 2 rows, got {self.rows} -- "
                "one row is a chain, so ask for geometry='chain'"
            )
        # The mirror of the row check, and it was missing. Rows were validated and
        # columns were not, so `TFIMSpec(n_sites=3, geometry="triangular", rows=3)`
        # was accepted here and then raised from `Lattice` the first time anything
        # touched `.lattice` -- a `describe_problem` call away from the mistake, by
        # which point the tool had already lost its chance to answer with the
        # `{"error": ...}` every other bad argument gets. A specification that
        # cannot be built is refused where it is written down.
        if self.geometry != "chain" and self.cols < MIN_SIDE:
            raise ValueError(
                f"a {self.geometry} lattice needs at least {MIN_SIDE} columns, and "
                f"{self.n_sites} sites in {self.rows} rows leaves {self.cols} -- "
                f"ask for at least {MIN_SIDE * self.rows} sites, or geometry='chain'"
            )

    @property
    def cols(self) -> int:
        """Columns of sites, which is the whole length for a chain.

        Derived from :attr:`n_sites` and :attr:`rows` rather than stored, so there
        is no second size to keep in step with the first.
        """
        return self.n_sites // self.rows

    @property
    def lattice(self) -> Lattice:
        """The shape as the geometry module expresses it.

        The one place a spec turns into an edge list, so every solver that takes a
        :class:`~src.physics.lattice.Lattice` is handed the same bonds for the same
        problem.

        Returns:
            The lattice. For a chain of two sites on a periodic boundary this
            raises instead, because :class:`~src.physics.lattice.Lattice` refuses
            that size -- see :data:`src.physics.lattice.TWO_SITE_RING`. A spec may
            still express it, and the solvers that generate their own chain bonds
            still answer it; only the edge-list route declines.

        Examples:
            >>> TFIMSpec(n_sites=6, boundary="open").lattice.bonds()
            ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))
            >>> TFIMSpec(n_sites=4, boundary="open", geometry="square", rows=2).lattice.n_bonds
            4
        """
        return Lattice(
            geometry=self.geometry,
            rows=self.rows,
            cols=self.cols,
            boundary=self.boundary,
        )

    @property
    def shape_to_solve(self) -> Lattice | None:
        """The geometry a solver should take its bonds from, or ``None`` for a line.

        Every solver in the project accepts an optional lattice and falls back to
        its own chain-bond generator when given none. This property is what decides
        between the two, and it answers ``None`` for a line **on purpose** rather
        than for convenience.

        A line's bonds are already generated two independent ways -- once in
        :func:`src.physics.quantum.hamiltonians.chain_bonds` and once in
        :func:`src.physics.reference.exact_diagonalisation.bonds` -- and those two
        routes agreeing is a check this project relies on. Routing a line through
        the shared edge list would replace two independent generators with one and
        quietly retire that check. It would also change one answer: an edge *set*
        cannot hold the doubled bond of a two-site ring, which three routes already
        agree about. See :data:`src.physics.lattice.TWO_SITE_RING`.

        Returns:
            The lattice for a two-dimensional problem, ``None`` for a line.

        Examples:
            >>> TFIMSpec(n_sites=8).shape_to_solve is None
            True
            >>> TFIMSpec(n_sites=16, geometry="square", rows=4).shape_to_solve.n_bonds
            32
        """
        return None if self.is_one_dimensional else self.lattice

    @property
    def is_one_dimensional(self) -> bool:
        """Whether this problem is a line, and so has a closed-form answer.

        The single predicate the rest of the project asks instead of comparing
        ``geometry`` to a string in a dozen places. A shape that has degenerated to
        one row counts as a line, because that is what it is.

        Examples:
            >>> TFIMSpec(n_sites=8).is_one_dimensional
            True
            >>> TFIMSpec(n_sites=16, geometry="square", rows=4).is_one_dimensional
            False
        """
        return self.geometry == "chain" or self.rows == 1

    @property
    def ratio(self) -> float:
        r"""The dimensionless control parameter ``h / J``.

        The quantum critical point of the infinite chain sits at ``h / J = 1``,
        where Kramers-Wannier self-duality pins it exactly.

        Written out as ``h / J`` and never as ``g``. Much of the transverse-field
        Ising literature calls this ratio ``g``, and that is unambiguous only in
        the papers that carry no longitudinal field. This project carries one --
        :data:`HAMILTONIAN_LATEX` has a ``-g \sum_i \hat\sigma^z_i`` term and the
        interface spends a slider on it -- so the letter is already taken, and a
        page that showed a ``g`` slider beside the words "g = h/J" would be showing
        one symbol for two quantities.
        """
        return self.field / self.coupling

    @property
    def n_bonds(self) -> int:
        """How many Ising bonds the coupling sum runs over.

        A line keeps the arithmetic it always had -- ``L`` for a ring, ``L - 1``
        for a segment -- rather than reading it off the edge list. That is
        deliberate: :meth:`src.physics.lattice.Lattice.bonds` builds a *set* of
        edges and so cannot express the doubled bond of a two-site ring, which this
        class is allowed to express and three solvers already agree about. See
        :data:`src.physics.lattice.TWO_SITE_RING`.

        A two-dimensional lattice has no such closed form, so its count comes from
        the edge list.

        Examples:
            >>> TFIMSpec(n_sites=6, boundary="periodic").n_bonds
            6
            >>> TFIMSpec(n_sites=6, boundary="open").n_bonds
            5
            >>> TFIMSpec(n_sites=16, boundary="open", geometry="square", rows=4).n_bonds
            24
        """
        if self.is_one_dimensional:
            return self.n_sites if self.boundary == "periodic" else self.n_sites - 1
        return self.lattice.n_bonds

    def label(self) -> str:
        """A short human-readable identifier, used in logs and plot legends.

        The shape appears only when it is not a line, so every label that existed
        before this field did reads exactly as it did -- which is what keeps a run
        and its grading line up by name across a change to the spec.

        Examples:
            >>> TFIMSpec(n_sites=6).label()
            'TFIM L=6 J=1 h=1 (peri)'
            >>> TFIMSpec(n_sites=16, geometry="square", rows=4).label()
            'TFIM 4x4 square L=16 J=1 h=1 (peri)'
        """
        shape = "" if self.is_one_dimensional else f" {self.rows}x{self.cols} {self.geometry}"
        return (
            f"TFIM{shape} L={self.n_sites} J={self.coupling:g} "
            f"h={self.field:g} ({self.boundary[:4]})"
        )
