"""Hand-built quantum layer: the circuits, their gradients, and their cost.

Everything under this package is written from the mathematics rather than
assembled from a library's template gallery, because the design decisions --
which ansatz, how deep, how the layers compile, how the shots are spent -- are
the substance of the project rather than an implementation detail of it.

This package may not see an exact answer. Nothing here may import
anything under :mod:`src.physics.reference` -- the Pfeuty closed form or the
sparse diagonalisation -- directly or transitively; the architecture test
walks the import graph and fails if it can. :mod:`src.physics.model` is
permitted and is the one exception -- it holds the problem *specification*, not
its solution, and the agent is of course allowed to know what it was asked.
"""
