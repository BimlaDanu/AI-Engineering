"""Repair a question's spelling against this project's own vocabulary.

The scope gate, the intent reader, the shelf router and the retrieval query all
work by matching words, so every one of them fails the same way on a typo. *What
is the phys of the quntum isng mdl?* shares not one word with
:data:`DOMAIN_TERMS`, so it is judged out of scope, no chain is formalised,
nothing is searched, and the reader is told the request could not be turned into
a physics problem. The question was clear; three characters were wrong.

Correction runs against this project's own vocabulary rather than a general
dictionary, which is what keeps it safe: the candidate set is the terms the
downstream matchers already know, so a repair can only move a word towards
something those matchers recognise.
"""

from __future__ import annotations

import re
from difflib import get_close_matches

from src.agent.intent import EXPLAIN_TERMS, FEASIBILITY_TERMS, IMPLEMENT_TERMS
from src.agent.router import DOMAIN_PHRASES, DOMAIN_TERMS, SHELF_TERMS

TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")
"""A word, keeping internal hyphens and apostrophes so ``free-fermion`` stays one."""

SHORTHAND: dict[str, str] = {
    "phys": "physics",
    "mdl": "model",
    "mdls": "models",
    "eqn": "equation",
    "eqns": "equations",
    "ham": "hamiltonian",
    "hamil": "hamiltonian",
    "qc": "quantum",
    "gs": "ground state",
    "tfim": "transverse field ising model",
    "temp": "temperature",
    "config": "configuration",
    "configs": "configurations",
    "params": "parameters",
    "param": "parameter",
    "approx": "approximation",
    "calc": "calculation",
    "sim": "simulation",
    "sims": "simulations",
    "algo": "algorithm",
    "algos": "algorithms",
    "opt": "optimisation",
    "dof": "degrees of freedom",
}
"""Contractions people type, and what they meant.

Listed rather than inferred because an abbreviation is not a misspelling. ``mdl``
scores 0.75 against ``model`` and ``phys`` scores 0.73 against ``physics`` -- both
below any cutoff that leaves ordinary words alone -- so a rule loose enough to catch
them would also rewrite words nobody mistyped. Naming the two dozen that occur in
practice costs nothing and is exactly as correct as the list is.
"""

ORDINARY: frozenset[str] = frozenset(
    """
    about above after again against all also always among and another answer any anything
    are around ask asked back because been before being below best better between both
    but can cannot come could data days describe did difference different does doing done
    down during each either else enough even ever every example explain few find first
    following for from full further get give given going good got had has have help her
    here him his how however into its itself just keep kind know large last later least
    less let like little long look made make many may mean means might more most much must
    need never new next not nothing now number often once one only other our out over own
    part people perhaps place please point possible put question quite rather really right
    said same say see seem set several she should show side since small some something
    still such sure take tell than that the their them then there these they thing things
    think this those though three through time times too two under until upon use used
    using very want was way well were what when where whether which while who whole why
    will with within without work would write yes yet you your
    axis chart charts curve curves draw drawn epoch epochs graph graphs loss losses
    plot plots plotted plotting versus
    """.split()
)
"""Ordinary English that must never be corrected towards a technical term.

The guard that makes the nearest-match rule safe. Without it ``plate`` scores 0.83
against ``plateau`` and ``states`` scores well against ``state``, so a question about
a dinner plate would come back about barren plateaus. A full dictionary would be
better and is not worth a dependency: the words at risk are the ones that appear in
questions, and those are short, common and enumerable.

The second line is the vocabulary of *asking for a picture*, and it is there because
``plot`` scored 0.89 against ``pilot`` -- a real word this project's hardware pages use
-- and so *plot the loss against epoch* was being restated as *pilot the loss*. That is
the exact failure this list exists to prevent, arriving through a word nobody thought
of as being at risk. See :data:`src.agent.reading.CURVE_WORDS`, which is what these
words are read for once they survive.
"""

MINIMUM_LENGTH = 4
"""Shortest word the nearest-match rule will consider.

Below four characters an edit distance says almost nothing -- ``sim``, ``gap`` and
``its`` are all within one edit of several unrelated terms -- and the abbreviations
worth catching at that length are in :data:`SHORTHAND` already, spelled out.
"""

CUTOFF = 0.82
"""How close a match has to be before a word is rewritten.

Tuned against the two cases that matter in both directions. ``quntum`` scores 0.92
against ``quantum`` and ``isng`` scores 0.89 against ``ising``, so real typos clear it
comfortably; the ordinary words that come nearest a technical term sit below it once
:data:`ORDINARY` has taken the closest ones out. Raising it further starts refusing
two-character slips, which are the commonest kind.
"""

LENGTH_SLACK = 1
"""How many characters a correction may add or remove.

A typo is a slip of one or two keys: a letter dropped, doubled, or hit twice. It
almost never changes a word's length by more than one, and insisting on that is what
stops the rule reaching across a real gap -- ``plate`` sits 0.83 from ``plateau``,
comfortably over :data:`CUTOFF`, and is two characters short of it. With this in
place a question about a dinner plate stays about a dinner plate.
"""

SUFFIXES = ("s", "es", "ed", "ing")
"""Endings stripped before deciding a word is unknown.

``states`` is not in the vocabulary and ``state`` is, and without this the rule
"corrects" the plural to the singular -- a change that is both wrong and invisible,
since the result is a real word in the right place. Checking the stem costs four
lookups and removes the entire class.
"""


def _known(word: str) -> bool:
    """Whether a word needs no correction at all.

    Args:
        word: The word, lowercased.

    Returns:
        ``True`` when it is vocabulary, ordinary English, or an inflection of
        either. An inflection is left exactly as the user wrote it: rewriting
        ``states`` to ``state`` is a silent change to a sentence that was correct.
    """
    if word in VOCABULARY or word in ORDINARY:
        return True
    if any(word.endswith(suffix) and word[: -len(suffix)] in VOCABULARY for suffix in SUFFIXES):
        return True
    # And the other direction. The vocabulary is assembled from matching lists, which
    # carry whichever form the matching wanted, so ``neighbours`` is in it and
    # ``neighbour`` is not -- and the singular, a correctly spelled English word, was
    # being "corrected" into the plural in the middle of the reader's own sentence.
    return any(word + suffix in VOCABULARY for suffix in SUFFIXES)


def _transposed(word: str) -> str | None:
    """Find a vocabulary word this one is a letter-shuffle of.

    The case edit-distance handles worst and typists produce most. ``feild`` and
    ``field`` share every letter and score 0.60 against each other, because
    :mod:`difflib` measures matching *blocks* and a swap breaks the word into three
    of them. Comparing sorted letters catches it exactly, and cannot fire by
    accident: an unknown word that is a precise anagram of a technical term is a
    mistyping of that term in every case that occurs in practice.

    Args:
        word: The word, lowercased.

    Returns:
        The vocabulary word, or ``None``.
    """
    return _ANAGRAMS.get("".join(sorted(word)))


_ANAGRAMS: dict[str, str] = {}
"""Vocabulary keyed by its sorted letters. Filled once, below."""


CORRECTABLE: frozenset[str] = frozenset(
    {
        "model",
        "models",
        "physics",
        "physical",
        "spectrum",
        "eigenvalues",
        "eigenstates",
        "qubits",
        "circuits",
        "algorithm",
        "algorithms",
        "temperature",
        "boundary",
        "periodic",
        "operator",
        "operators",
        "expectation",
        "wavefunction",
        "degenerate",
        "degeneracy",
        "hardware",
        "accuracy",
        "convergence",
        "parameter",
        "parameters",
        "gradient",
        "gradients",
        "optimiser",
        "optimizer",
        "benchmark",
        "noise",
        "error",
        "errors",
    }
)
"""Words worth repairing towards that must not, on their own, open the scope gate.

The two jobs are different and were being done by one list. ``model`` is plainly a
word this project's questions contain -- *what is the phys of the quntum isng mdl?*
is unanswerable without it -- but a question containing only "model" is as likely to
be about a machine-learning model or a fashion model as about this one, so putting it
in :data:`~src.agent.router.DOMAIN_TERMS` would widen the gate rather than the
spelling. Keeping the correction vocabulary a superset of the gate's vocabulary lets
each be exactly as generous as its own job requires.
"""


def vocabulary() -> frozenset[str]:
    """Every word this project matches questions against.

    Assembled from the lists the matching actually uses -- the scope gate, the shelf
    router and the three intent readers -- rather than copied into a fourth list
    here. A vocabulary that drifts from the gate it is meant to open is worse than no
    correction at all: it would repair a word into something still out of scope.

    Returns:
        The union, single words only. Multi-word entries are split, because the
        repair works one token at a time. :data:`CORRECTABLE` is folded in for the
        words that belong in a question here but should not open the scope gate by
        themselves.
    """
    words: set[str] = set(DOMAIN_TERMS) | set(CORRECTABLE)
    words |= {word for phrase in DOMAIN_PHRASES for word in phrase.split()}
    for shelf in SHELF_TERMS.values():
        words |= {word for term in shelf for word in term.split()}
    for terms in (EXPLAIN_TERMS, FEASIBILITY_TERMS, IMPLEMENT_TERMS):
        words |= {word for term in terms for word in term.split()}
    return frozenset(words)


VOCABULARY = vocabulary()
"""The vocabulary, built once. See :func:`vocabulary`."""


_ANAGRAMS.update(
    # Last one wins on a collision, which is harmless: two vocabulary words that are
    # anagrams of each other are two words either of which is a fair reading of a
    # shuffle of them, and there is no such pair in this vocabulary today.
    {"".join(sorted(word)): word for word in sorted(VOCABULARY)}
)


def _repair(word: str) -> str:
    """Correct one word, or leave it exactly as it is.

    Args:
        word: The word as typed, in any case.

    Returns:
        The correction, or the word unchanged. Unchanged is the default and every
        rule below is a reason to return early with it.
    """
    lowered = word.lower()
    if lowered in SHORTHAND:
        return _cased_like(SHORTHAND[lowered], word)
    if _known(lowered):
        return word
    if len(lowered) < MINIMUM_LENGTH:
        return word
    swapped = _transposed(lowered)
    if swapped is not None:
        return _cased_like(swapped, word)
    # Several candidates rather than one, then the nearest of those that is also the
    # right length. Asking for a single match and rejecting it would throw away the
    # correct answer whenever a shorter word happened to score marginally higher --
    # `circuts` is nearer `circuits` than `circuit`, and only just.
    close = [
        candidate
        for candidate in get_close_matches(lowered, VOCABULARY, n=5, cutoff=CUTOFF)
        if abs(len(candidate) - len(lowered)) <= LENGTH_SLACK
    ]
    return _cased_like(close[0], word) if close else word


def _cased_like(correction: str, typed: str) -> str:
    """Give a correction the capitalisation of the word it replaces.

    The vocabulary is lowercase, so a repair returned it lowercase and a word that
    began a sentence lost its capital -- which downstream readers use to find where
    one sentence ends and the next begins.

    Args:
        correction: The repaired word, lowercase.
        typed: The word as the user wrote it.

    Returns:
        The correction, capitalised or upper-cased to match.
    """
    if typed.isupper() and len(typed) > 1:
        return correction.upper()
    if typed[:1].isupper():
        return correction[:1].upper() + correction[1:]
    return correction


def repair(question: str) -> str:
    """Rewrite a question's domain vocabulary into the spelling this project matches.

    Args:
        question: The question exactly as it was typed.

    Returns:
        The question with recognised misspellings and abbreviations corrected, and
        everything else -- punctuation, casing of untouched words, spacing --
        preserved. Identical to the input when nothing was recognised, which is the
        common case and the one that must cost nothing.

    Examples:
        >>> repair("what is Phys of quntum isng mdl?")
        'what is Physics of quantum ising model?'
        >>> repair("the neighbour interaction")
        'the neighbour interaction'
        >>> repair("How deep must the ansatz circuit be?")
        'How deep must the ansatz circuit be?'
        >>> repair("What is the best pizza in Vilnius?")
        'What is the best pizza in Vilnius?'
    """
    return TOKEN.sub(lambda found: _repair(found.group(0)), question)


def corrections(question: str) -> tuple[tuple[str, str], ...]:
    """List the words that would be changed, for a trace and for a test.

    Reported rather than only applied. A correction is a guess about what somebody
    meant, and a guess nobody can see is the kind that gets believed -- a reader who
    can read *"quntum → quantum"* can tell at once whether the agent understood them.

    Args:
        question: The question exactly as it was typed.

    Returns:
        Each ``(as typed, as read)`` pair, in the order they appear, without
        duplicates.

    Examples:
        >>> corrections("what is Phys of quntum isng mdl?")
        (('Phys', 'Physics'), ('quntum', 'quantum'), ('isng', 'ising'), ('mdl', 'model'))
    """
    seen: dict[str, str] = {}
    for found in TOKEN.finditer(question):
        word = found.group(0)
        fixed = _repair(word)
        if fixed != word and word not in seen:
            seen[word] = fixed
    return tuple(seen.items())
