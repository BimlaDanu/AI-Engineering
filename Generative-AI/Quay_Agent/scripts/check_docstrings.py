"""Fail the build when a docstring turns into an essay.

Ruff's `D` rules check that a docstring exists and is shaped like Google style,
not that it is short. Four mechanical defects are flagged: prose lines over
budget, a reST section heading, a changelog phrase, and markdown bold.

    uv run python -m scripts.check_docstrings [--path src/agent] [--quiet]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

GOOGLE_SECTION = re.compile(
    r"^\s*(Args|Arguments|Returns|Yields|Raises|Attributes|Examples?|Example"
    r"|Note|Notes|Warning|Warnings|Todo|References)\s*:\s*$"
)
"""Start of a Google-style block, whose body is specification and not prose."""

REST_UNDERLINE = re.compile(r"^\s*[-=~^\"'#*+]{3,}\s*$")
"""A reST section heading's underline, and the tell-tale of an essay."""

DIRECTIVE = re.compile(r"^(?P<indent>\s*)\.\.\s+\w[\w-]*::")
"""A reST directive -- ``.. math::`` and friends -- whose body is content.

A displayed equation is the thing the docstring is for, not narrative about it,
so the directive and its indented body are not rationed.
"""

LIST_ITEM = re.compile(r"^(?P<indent>\s*)(?:[-*+]|\d+[.)])\s+\S")
"""A bullet or numbered item, which is structure rather than narrative.

An enumeration of a module's parts is a map, and a map earns its place however
many entries it has. Only the prose around it is rationed.
"""

TABLE_BORDER = re.compile(r"^\s*[-=]{2,}(?:\s+[-=]{2,})+\s*$")
"""A simple-table rule, which is a column layout rather than a heading.

A table of contents earns its place in a package docstring; a section heading is
what prose grows when there is too much of it. The two look alike one line at a
time, and only the border with more than one run is the table.
"""

CHANGELOG_PHRASE = re.compile(
    r"(?<!\bis )(?<!\bare )(?<!\bwas )(?<!\bwere )(?<!\bbe )(?<!\bbeen )"
    r"(?<!\bbeing )(?<!\bnot )\bused to\b"
    r"|\b(?:previously|formerly|originally)\b"
    r"|\ban earlier (?:version|implementation|iteration|revision|build|state class)\b"
    r"|\bthe (?:first|original) (?:version|implementation|cut)\b"
    r"|\bbefore (?:this|that)(?: \w+)? existed\b"
    r"|\bwas rewritten\b|\bhas since been\b|\bused to be\b",
    re.IGNORECASE,
)
"""Prose about the code's history, which git already records.

Purpose and history share a grammar -- "is used to break a tie" against "used to
break a tie" -- so the passive is excluded by lookbehind. A present-tense fact
about what the code does not do ("a level this version no longer has") is not
history and is not matched.
"""

BOLD_LEAD_IN = re.compile(r"^\s*\*\*(?P<claim>[^*]{12,})\*\*", re.MULTILINE)
"""A bold sentence opening a paragraph, which is a heading in disguise.

Bold quoting a short user-facing label -- a button, a verdict, a tab -- is naming
a string the reader will see, so only spans of a dozen characters or more that
start a line are flagged.
"""

MODULE_PROSE_LIMIT = 20
"""Prose lines a module docstring may spend before it reads as an essay.

One screen. Enough for a module with several parts to introduce each of them,
and not enough to argue a case.
"""

DEFINITION_PROSE_LIMIT = 24
"""The same budget for a class or function, which may carry a worked example."""

Documentable = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
"""The node kinds `ast.get_docstring` accepts."""


@dataclass(frozen=True)
class Finding:
    """One docstring defect, addressed the way a compiler error is.

    Attributes:
        path: File the docstring lives in.
        line: Line the definition starts on.
        name: `module`, or the class or function name.
        reason: What is wrong, in one phrase.
        kind: Machine-readable tag, for the summary tally.
    """

    path: Path
    line: int
    name: str
    reason: str
    kind: str

    def __str__(self) -> str:
        """Render as an editor-clickable line.

        Returns:
            `path:line: name: reason`.
        """
        return f"{self.path}:{self.line}: {self.name}: {self.reason}"


def prose_lines(docstring: str) -> int:
    """Count the narrative lines in a docstring.

    Google-style blocks and doctests are skipped: a long `Args:` list is a wide
    signature, not an essay, and penalising it would push authors to document
    less.

    Args:
        docstring: The raw docstring, newlines intact.

    Returns:
        How many lines carry prose.
    """
    total = 0
    in_block = False
    in_table = False
    list_indent: int | None = None
    directive_indent: int | None = None
    for line in docstring.splitlines():
        if found := DIRECTIVE.match(line):
            directive_indent = len(found.group("indent"))
            continue
        if directive_indent is not None:
            if not line.strip() or len(line) - len(line.lstrip()) > directive_indent:
                continue
            directive_indent = None
        if item := LIST_ITEM.match(line):
            list_indent = len(item.group("indent"))
            continue
        if list_indent is not None:
            # A wrapped item is indented past its own marker; anything else ends the list.
            if not line.strip() or len(line) - len(line.lstrip()) > list_indent:
                continue
            list_indent = None
        if TABLE_BORDER.match(line):
            in_table = True
            continue
        if in_table:
            # A simple table has three borders, so a blank line ends it, not a toggle.
            in_table = bool(line.strip())
            continue
        if GOOGLE_SECTION.match(line):
            in_block = True
            continue
        stripped = line.strip()
        if in_block:
            if not stripped or line[:1].isspace():
                continue
            in_block = False
        if not stripped or stripped.startswith((">>>", "...")):
            continue
        total += 1
    return total


def documented(tree: ast.Module) -> list[tuple[str, Documentable]]:
    """Every node in one file that can carry a docstring.

    Args:
        tree: The parsed module.

    Returns:
        Pairs of display name and node, the module first.
    """
    found: list[tuple[str, Documentable]] = [("module", tree)]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            found.append((node.name, node))
    return found


def inspect(path: Path) -> list[Finding]:
    """Check one file's docstrings.

    Args:
        path: The module to read.

    Returns:
        Every finding in the file, in source order.
    """
    findings: list[Finding] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for name, node in documented(tree):
        docstring = ast.get_docstring(node, clean=False)
        if docstring is None:
            continue
        line = getattr(node, "lineno", 1)
        limit = MODULE_PROSE_LIMIT if name == "module" else DEFINITION_PROSE_LIMIT
        spent = prose_lines(docstring)
        if spent > limit:
            findings.append(
                Finding(path, line, name, f"{spent} prose lines, over {limit}", "essay")
            )
        underlines = sum(
            bool(REST_UNDERLINE.match(row)) and not TABLE_BORDER.match(row)
            for row in docstring.splitlines()
        )
        if underlines:
            findings.append(
                Finding(path, line, name, f"{underlines} reST section heading(s)", "heading")
            )
        if match := CHANGELOG_PHRASE.search(docstring):
            findings.append(
                Finding(path, line, name, f"changelog prose: {match.group(0)!r}", "changelog")
            )
        if bold := BOLD_LEAD_IN.search(docstring):
            claim = " ".join(bold.group("claim").split())
            findings.append(Finding(path, line, name, f"bold lead-in: {claim[:48]}", "bold"))
    findings.sort(key=lambda found: found.line)
    return findings


def main() -> int:
    """Check every module under the requested path.

    Returns:
        Process exit status: 0 when clean, 1 on any finding.
    """
    parser = argparse.ArgumentParser(description="Flag docstrings that grew into essays.")
    parser.add_argument("--path", default="src", help="directory or file to check")
    parser.add_argument("--quiet", action="store_true", help="print the tally only")
    arguments = parser.parse_args()

    root = Path(arguments.path)
    modules = sorted(root.rglob("*.py")) if root.is_dir() else [root]

    findings: list[Finding] = []
    for module in modules:
        findings.extend(inspect(module))

    if not arguments.quiet:
        for finding in findings:
            print(finding)
        print()

    tally = Counter(finding.kind for finding in findings)
    if not findings:
        print(f"{len(modules)} modules checked, every docstring within budget")
        return 0

    breakdown = ", ".join(f"{count} {kind}" for kind, count in sorted(tally.items()))
    per_file = Counter(str(finding.path) for finding in findings)
    print(f"{len(findings)} findings in {len(per_file)} of {len(modules)} modules: {breakdown}")
    for path, count in per_file.most_common(10):
        print(f"  {count:>3}  {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
