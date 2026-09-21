r"""The device layer: what a real machine can actually be asked to run.

Everything under :mod:`src.physics.quantum` prices a circuit in the abstract --
so many :math:`\hat\sigma^z\hat\sigma^z` rotations, so many layers, so many
parameters. None of that is what a machine executes. A real processor has a fixed
set of qubits wired to *some* of their neighbours and not the rest, a native gate
that is probably not the one the algorithm was written in, a duration for every
operation, and a window of coherence that closes whether or not the circuit has
finished. This package is the translation between the two, and the arithmetic that
says whether the translated circuit still means anything when it lands.

Why this is separated from the circuit layer
--------------------------------------------

An ansatz costs what it costs; what it costs *here* depends on which machine is
being asked. The same twelve-site chain is free of routing on a line and needs
seventeen extra two-qubit layers on a lattice that has no twelve-qubit path -- and
the difference is the whole feasibility question, not a detail of it. Keeping the
device out of :class:`~src.physics.quantum.ansatz.AnsatzSpec` is what lets one
specification be priced against three machines without being rebuilt, and what
lets the ideal machine exist at all as the control that isolates algorithmic error
from hardware error.

No vendor SDK
-------------

The connectivity, the durations and the error rates are transcribed here as data,
and every stage below operates on that data with integer arithmetic and a
breadth-first search. That is a deliberate choice rather than an omission. A
transpiler shipped by a vendor is a large dependency whose routing heuristic
changes between releases, and a depth number that moves when a package is upgraded
cannot support a verdict. What is written here is worse at routing than a
production compiler and it is stated in one place, in full, where a reader can
disagree with it.

======================  =====================================================
Module                  What it holds
======================  =====================================================
``devices.py``          the machines: connectivity, durations, error rates
``transpile.py``        decompose, place, route, schedule
``fidelity.py``         what survives the noise, and the depth ceiling it sets
``export.py``           the circuit and the run card, as files somebody can submit
======================  =====================================================

Nothing here may reach an exact solution. A device model that could look up the
ground-state energy would make every fidelity estimate in the package unfalsifiable,
and the import graph is checked for it.
"""
