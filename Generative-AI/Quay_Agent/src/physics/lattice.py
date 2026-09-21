r"""Which sites are coupled to which -- the geometry, and nothing else.

Given a shape, which pairs of sites interact? The answer is an edge list, which
gives nothing away about an energy, so every layer may read this module.

Geometry lives apart so that a second lattice is a change of data rather than of
code. Both halves of the project build operators from an edge list --
:func:`src.physics.quantum.hamiltonians.chain_bonds` on the agent\'s side,
:func:`src.physics.reference.exact_diagonalisation.bonds` on the grader\'s -- and
they stay separate implementations, because two routes sharing a bond generator
share its bugs.

Three shapes:

``chain``
    One dimension, and the only shape with a closed form: Jordan-Wigner
    linearises a chain and nothing else, so
    :mod:`src.physics.reference.free_fermions` applies here alone.

``square``
    Two dimensions, bipartite. On a bipartite lattice a ferromagnet and an
    antiferromagnet have identical spectra, related by flipping every spin on one
    sublattice, so ``J`` and ``-J`` must agree to machine precision -- a free
    cross-check, and a disagreement is this project\'s bug.

``triangular``
    Two dimensions, not bipartite: its triangles cannot be two-coloured. With
    antiferromagnetic coupling the three bonds of a triangle cannot all be
    satisfied at once, which is frustration, and it is the one regime in reach
    where the classical competition genuinely weakens.

    Frustration here is not a sign problem. This model is diagonal in the
    :math:`\sigma^z` basis apart from the transverse field, so every off-diagonal
    element is :math:`-h` -- negative on any graph, for either sign of :math:`J`
    -- and non-positive off-diagonal elements give non-negative path-integral
    weights. Frustration moves the diagonal, which no weight\'s sign depends on;
    the regime test asserts this from assembled matrices on
    all three shapes. What it costs instead is slow mixing, so a sampler\'s
    successive measurements stay correlated and its reported error bar is
    smaller than the real one. :mod:`src.physics.classical.regime` refuses a
    quantum-lead claim on that basis.

Bipartiteness is computed, not declared: :meth:`Lattice.is_bipartite`
two-colours the graph, so the cross-check above tests the edge list rather than
the label somebody typed.

One calibration anchor. A ``2 x 2`` open square is a four-cycle, and so is a
four-site ring -- isomorphic but not identical, since the square runs
``0-1-3-2-0`` under row-major numbering and the ring ``0-1-2-3-0``. An energy is
invariant under relabelling, so the 2D path at its smallest size must reproduce
the 1D closed form exactly: at ``J = h = 1`` both give ``-5.226251860``.
The lattice test checks the shape and the number separately.

It is the only such identity available. A single triangle would be a three-site
ring, but a two-dimensional lattice needs two rows and two columns, so the
smallest triangular lattice here has four sites and is not a ring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Geometry = Literal["chain", "square", "triangular"]
"""The shapes a problem may be posed on.

Three, not a general graph. A lattice a person can name is a lattice they can picture,
and every question this application answers is about one of these -- so an arbitrary
edge list would be a feature nobody could ask for in a sentence, which is the test
:mod:`src.agent.reading` has to pass.
"""

Boundary = Literal["open", "periodic"]
"""Whether the edges of the lattice wrap around.

``periodic`` removes the boundary and is the closer approximation to an infinite
system at a given size; ``open`` is what a real device's wiring more often looks
like. Both are offered because the difference is measurable and interesting, not
because one is right.
"""

DEFAULT_SIDES: dict[Geometry, tuple[int, int]] = {
    "chain": (1, 12),
    "square": (2, 2),
    "triangular": (3, 3),
}
"""Size a geometry is built at when a question names a shape and no size.

**Deliberately the smallest useful cluster of each shape, not the largest runnable
one.** A ``2 x 2`` square is four sites and a ``3 x 3`` triangular cluster is nine,
both of which any method here answers in milliseconds -- so a question that names a
shape and nothing else is answered immediately rather than starting a minute of
arithmetic nobody asked for.

The measured ceilings are much higher and are a separate decision from the default:
energy-and-gradient on a state vector costs 4 ms at twelve sites, 47 ms at sixteen
and 1.8 s at twenty, and sparse exact diagonalisation of a ``4 x 4`` square (sixteen
sites, twenty-four bonds) takes 171 ms. So a ``4 x 4`` square and a twelve-site
triangular cluster are both comfortable when *asked for*. They are not what an
unspecified question gets.

The chain keeps the length it always had, so nothing about existing behaviour moves.
"""

MIN_SIDE = 2
"""Smallest side length a two-dimensional lattice may have.

One row is not two-dimensional -- it is a chain, and :meth:`Lattice.bonds` returns
exactly the chain's bonds for it. Refusing it here would be pedantry; what would be
wrong is *calling* it a square lattice in a report, so :meth:`Lattice.describe` says
"chain" when a shape degenerates to one.
"""

TWO_SITE_RING = 2
r"""The one size at which *this module* refuses a periodic boundary.

Found by the geometry tests in Phase 0 of the 2D work. The project's bond generators
disagree about a two-site ring:

- :func:`src.physics.quantum.hamiltonians.chain_bonds` returns ``((0, 1), (1, 0))``,
  which is :math:`\sum_i \sigma^z_i \sigma^z_{i+1 \bmod L}` written out literally --
  the same coupling twice, so :math:`2J`.
- :meth:`Lattice.bonds` returns ``((0, 1),)``, because it is an edge set and a graph
  has one edge there.

Those are **different energies**, so it is a real disagreement rather than a
notational one, and Phase 2 wires both to the same geometry -- which would otherwise
leave the project's central claim, that two independent routes agree, with one size
where it silently does not.

**The doubled reading wins, and it is not a coin-toss.** The registry test
runs ``L = 2`` periodic through the closed form and the grader's sparse solver and
they agree there, so three routes already concur and are validated against Pfeuty.
An earlier version of this change refused the size in
:class:`src.physics.model.TFIMSpec` as well, on the reasoning that a two-site ring is
degenerate and nobody wants one. That was wrong: it deleted working, tested coverage
in the name of protecting a claim the tests had already settled, and those two
registry tests failed and said so.

So the refusal belongs here alone. :meth:`Lattice.bonds` builds a set of edges and
*cannot* express a doubled one, so this is the single route that would get the size
wrong, and declining it is the honest thing for it to do. Ask through
:class:`~src.physics.model.TFIMSpec` for a two-site ring and it still works.

Numerically equal to :data:`MIN_SIDE` and kept separate on purpose -- one is the
smallest side a *lattice* may have, the other the size at which a *ring* is refused,
and a later change to either must not silently move the other.
"""


@dataclass(frozen=True, slots=True)
class Lattice:
    """A shape, a size, and whether its edges wrap.

    Frozen so it can be hashed into a run cache: two identical lattices must always
    produce the same bond list, and therefore the same answer.

    Attributes:
        geometry: Which shape. See :data:`Geometry`.
        rows: Rows of sites. For a chain this is ignored and the length is
            :attr:`cols`.
        cols: Columns of sites, and the whole length for a chain.
        boundary: Whether the edges wrap. See :data:`Boundary`.
    """

    geometry: Geometry = "chain"
    rows: int = 1
    cols: int = 12
    boundary: Boundary = "open"

    def __post_init__(self) -> None:
        """Reject a size that does not describe the shape it claims to be.

        Raises:
            ValueError: If a side is too small, or a two-dimensional lattice was
                asked for with a single row -- which is a chain wearing another
                shape's name, and the report would then describe a lattice nobody
                asked about.
        """
        if self.cols < MIN_SIDE:
            raise ValueError(f"cols must be at least {MIN_SIDE}, got {self.cols}")
        if self.geometry == "chain":
            if self.rows != 1:
                raise ValueError(f"a chain has one row, got {self.rows}")
            if self.cols == TWO_SITE_RING and self.boundary == "periodic":
                raise ValueError(
                    "a two-site ring is refused; see TWO_SITE_RING for why. Ask for "
                    "boundary='open' instead"
                )
            return
        if self.rows < MIN_SIDE:
            raise ValueError(
                f"a {self.geometry} lattice needs at least {MIN_SIDE} rows, got "
                f"{self.rows} -- one row is a chain, so ask for one"
            )

    @property
    def n_sites(self) -> int:
        """How many sites the lattice holds.

        Examples:
            >>> Lattice("chain", 1, 12).n_sites
            12
            >>> Lattice("square", 4, 4).n_sites
            16
            >>> Lattice("triangular", 3, 4).n_sites
            12
        """
        return self.rows * self.cols

    def site(self, row: int, col: int) -> int:
        """Index of the site at a row and column, wrapping if periodic.

        Args:
            row: Which row, counted from zero. Wrapped modulo :attr:`rows`.
            col: Which column, counted from zero. Wrapped modulo :attr:`cols`.

        Returns:
            The flat index, row-major. Row-major throughout, and stated here because
            a bond list is only meaningful alongside the indexing that produced it --
            two modules disagreeing about it would produce two different lattices
            that both looked right.
        """
        return (row % self.rows) * self.cols + (col % self.cols)

    def bonds(self) -> tuple[tuple[int, int], ...]:
        """Every coupled pair, each listed once, in a stable order.

        Each pair exactly once and always as ``(lower, higher)``: a bond counted
        twice would double that term in the Hamiltonian, which is the mistake a
        periodic ``2 x 2`` lattice invites -- wrapping in both directions there
        connects each neighbour pair twice, so the wrap is skipped when a side is
        only two sites long.

        Returns:
            The edge list, sorted. Sorted rather than merely deterministic, because
            :func:`src.physics.quantum.ansatz.bond_rounds` colours this graph to
            schedule gates and a reordering would silently change the circuit's
            depth.

        Examples:
            A chain is what it always was:

            >>> Lattice("chain", 1, 4).bonds()
            ((0, 1), (1, 2), (2, 3))
            >>> Lattice("chain", 1, 4, "periodic").bonds()
            ((0, 1), (0, 3), (1, 2), (2, 3))

            A two-by-two open square is a four-cycle -- the calibration anchor:

            >>> Lattice("square", 2, 2).bonds()
            ((0, 1), (0, 2), (1, 3), (2, 3))

            The smallest triangular lattice adds one diagonal to that square,
            which is what closes a triangle and what makes it non-bipartite:

            >>> Lattice("triangular", 2, 2).bonds()
            ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3))
        """
        found: set[tuple[int, int]] = set()
        wrap_rows = self.boundary == "periodic" and self.rows > MIN_SIDE
        wrap_cols = self.boundary == "periodic" and self.cols > MIN_SIDE
        for row in range(self.rows):
            for col in range(self.cols):
                here = self.site(row, col)
                for step_row, step_col in self._neighbour_steps():
                    next_row, next_col = row + step_row, col + step_col
                    if next_row >= self.rows and not wrap_rows:
                        continue
                    if next_col >= self.cols and not wrap_cols:
                        continue
                    if next_col < 0 and not wrap_cols:
                        continue
                    there = self.site(next_row, next_col)
                    if there != here:
                        found.add((min(here, there), max(here, there)))
        return tuple(sorted(found))

    def _neighbour_steps(self) -> tuple[tuple[int, int], ...]:
        """The offsets that reach each site's forward neighbours.

        Forward only -- each bond is generated from one of its two ends -- so the
        set stays small and no pair is produced twice.

        Returns:
            ``(row_step, col_step)`` offsets for this geometry. The triangular
            lattice is a square one plus one diagonal, which is what makes its
            plaquettes triangles and what makes it non-bipartite.
        """
        if self.geometry == "chain":
            return ((0, 1),)
        if self.geometry == "square":
            return ((0, 1), (1, 0))
        return ((0, 1), (1, 0), (1, -1))

    @property
    def n_bonds(self) -> int:
        """How many coupled pairs there are."""
        return len(self.bonds())

    @property
    def is_bipartite(self) -> bool:
        """Whether the sites two-colour, so that every bond joins the colours.

        Computed by colouring the edge list, never read off the geometry's name.
        A ferromagnet and an antiferromagnet have identical
        spectra on a bipartite lattice and different ones otherwise, so this
        property is what the cross-check in the lattice test asserts
        against. Reading it from a label would make that test a tautology.

        Returns:
            True when a two-colouring exists. Frustration is possible only when
            this is false and the coupling is antiferromagnetic.

        Examples:
            >>> Lattice("chain", 1, 6).is_bipartite
            True
            >>> Lattice("square", 4, 4).is_bipartite
            True
            >>> Lattice("triangular", 3, 4).is_bipartite
            False
            >>> Lattice("chain", 1, 5, "periodic").is_bipartite
            False
        """
        neighbours: dict[int, list[int]] = {site: [] for site in range(self.n_sites)}
        for left, right in self.bonds():
            neighbours[left].append(right)
            neighbours[right].append(left)
        colour: dict[int, int] = {}
        for start in range(self.n_sites):
            if start in colour:
                continue
            colour[start] = 0
            queue = [start]
            while queue:
                here = queue.pop()
                for there in neighbours[here]:
                    if there not in colour:
                        colour[there] = 1 - colour[here]
                        queue.append(there)
                    elif colour[there] == colour[here]:
                        return False
        return True

    def frustrated_by(self, coupling: float) -> bool:
        """Whether this lattice frustrates a coupling of this sign.

        Args:
            coupling: The Ising coupling ``J``. Negative is antiferromagnetic in
                this project's convention, where the coupling term is written
                ``-J sum sigma^z sigma^z``.

        Returns:
            True only for an antiferromagnetic coupling on a lattice that does not
            two-colour. Frustration needs both: a ferromagnet is satisfied by every
            spin agreeing, whatever the shape, and a bipartite antiferromagnet is
            satisfied by alternating.

        Examples:
            >>> Lattice("triangular", 3, 4).frustrated_by(-1.0)
            True
            >>> Lattice("triangular", 3, 4).frustrated_by(1.0)
            False
            >>> Lattice("square", 4, 4).frustrated_by(-1.0)
            False
        """
        return coupling < 0.0 and not self.is_bipartite

    def describe(self) -> str:
        """Name the shape and size in the words a report should use.

        Returns:
            A short label. A two-dimensional lattice that has degenerated to a
            single row is called a chain, because that is what it is and a report
            calling it a square lattice would be describing a problem nobody asked
            about.
        """
        edge = "periodic" if self.boundary == "periodic" else "open"
        if self.geometry == "chain" or self.rows == 1:
            return f"chain of {self.n_sites} ({edge})"
        return f"{self.rows}x{self.cols} {self.geometry} ({edge})"

    def in_words(self) -> str:
        """Describe the shape in a sentence a reader with no physics can follow.

        :meth:`describe` is a label for a log line or a figure legend. This is the
        version for prose, and the two are kept apart because a report that reads
        "the problem is a 4x4 square (open)" has handed its reader a field from a
        dataclass rather than a description of anything.

        Returns:
            A noun phrase, without a leading article and without a full stop, so a
            caller can put it wherever the sentence needs it.

        Examples:
            >>> Lattice("chain", 1, 12).in_words()
            'line of 12 spins with two ends'
            >>> Lattice("chain", 1, 12, "periodic").in_words()
            'ring of 12 spins'
            >>> Lattice("square", 4, 4).in_words()
            '4 by 4 square lattice of 16 spins with open edges'
            >>> Lattice("triangular", 3, 4, "periodic").in_words()
            '3 by 4 triangular lattice of 12 spins, wrapped at every edge'
        """
        if self.geometry == "chain" or self.rows == 1:
            if self.boundary == "periodic":
                return f"ring of {self.n_sites} spins"
            return f"line of {self.n_sites} spins with two ends"
        shape = f"{self.rows} by {self.cols} {self.geometry} lattice of {self.n_sites} spins"
        if self.boundary == "periodic":
            return f"{shape}, wrapped at every edge"
        return f"{shape} with open edges"

    def neighbours_per_site(self) -> int:
        """How many neighbours a site in the interior has.

        The number that decides how much harder a shape is than a line, and it is
        the honest one-number answer to "why is 2D expensive?". Every extra
        neighbour is another two-qubit gate per site in the circuit and another
        term per site in the Hamiltonian.

        Returns:
            The interior coordination number: two for a line, four for a square
            lattice, six for a triangular one. Sites on an open boundary have
            fewer, which is why this says "in the interior" rather than reporting
            an average nobody asked for.

        Examples:
            >>> Lattice("chain", 1, 8).neighbours_per_site()
            2
            >>> Lattice("square", 4, 4).neighbours_per_site()
            4
            >>> Lattice("triangular", 4, 4).neighbours_per_site()
            6
        """
        return 2 * len(self._neighbour_steps())
