"""The reference answers. Nothing the agent runs may import from this package.

Two independent exact solutions of the transverse-field Ising chain, kept behind
one import prefix so that the rule guarding them is a single line rather than a
list that someone will forget to extend.

- :mod:`~src.physics.reference.free_fermions` -- the Pfeuty closed form, reached
  by Jordan-Wigner and a Bogoliubov rotation. No matrix is ever built, and the
  cost does not grow with chain length.
- :mod:`~src.physics.reference.exact_diagonalisation` -- a sparse matrix in the
  spin basis, diagonalised directly. Exponential in the chain length and
  therefore size-capped, and it shares no algebra at all with the route above.

Agreement between two methods with nothing in common is evidence; a passing
self-consistency check inside one method is not. That is the project's thesis,
and this package is where it is applied to the project's own answers.

The seal matters because the agent is graded by comparing its answers with
these. If any module the agent runs could read them, the comparison would prove
nothing: the agent might have looked the answer up, and no amount of good
behaviour elsewhere would say which had happened.
The architecture test walks the import graph of every agent-facing
package and fails if this prefix is reachable, directly or transitively.
:mod:`src.physics.model` is deliberately *not* here -- it is the problem
statement, and the agent is of course allowed to know what it was asked.
"""
