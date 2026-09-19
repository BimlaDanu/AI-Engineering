"""Conversation export: turn the chat history into JSON, CSV, or a PDF transcript.

These are pure functions over ``st.session_state.history`` — a list of message dicts as
built by :func:`src.ui.pages.chat.process_question` — so they are unit-testable without
Streamlit and without a network. Assistant messages carry rich metadata (``meta`` with
model/level/tokens/cost and a ``sources`` list); user messages carry only ``content``. Each
exporter degrades gracefully when those optional fields are absent (e.g. an early-exit
refusal has no ``meta``).

The PDF path uses ``fpdf2``. Its core fonts are Latin-1 only, so text is passed through
:func:`_pdf_safe`, which maps common typographic/maths glyphs to ASCII and drops anything
still unrepresentable — a readable transcript rather than a faithful render of every emoji.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any

# Common non-Latin-1 glyphs that show up in answers, mapped to ASCII lookalikes so the PDF
# stays readable instead of peppered with "?". Anything not covered is dropped by _pdf_safe.
_GLYPHS = {
    "—": "-",
    "–": "-",  # em / en dash
    "‘": "'",
    "’": "'",  # curly single quotes
    "“": '"',
    "”": '"',  # curly double quotes
    "…": "...",  # ellipsis
    "•": "-",
    "·": "-",  # bullet / middle dot
    "≈": "~",
    "≠": "!=",  # approx / not-equal
    "≤": "<=",
    "≥": ">=",  # le / ge
    "×": "x",
    "÷": "/",  # times / divide
    "→": "->",
    "←": "<-",  # arrows
    "±": "+/-",
    "√": "sqrt",  # plus-minus / root
}


@dataclass
class TurnRecord:
    """One normalised conversation turn, flattened for tabular/PDF export."""

    index: int
    role: str
    content: str
    model: str | None = None
    level: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost: float | None = None
    sources: list[str] = field(default_factory=list)


# Leading characters that spreadsheet apps (Excel / Google Sheets / LibreOffice) interpret as
# the start of a formula. A cell beginning with one of these can execute on open — a formula or
# DDE-injection attack that lands on whoever opens the exported file, not just the operator. We
# prefix such values with a single quote (OWASP CSV-injection guidance), which renders them as
# literal text. Applied to every free-text cell that can carry user- or model-authored content.
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value: str) -> str:
    """Neutralise spreadsheet formula/DDE injection by quoting a risky leading character."""
    return "'" + value if value[:1] in _CSV_FORMULA_PREFIXES else value


def _source_lines(sources: list[dict[str, Any]]) -> list[str]:
    """Render each source dict as a compact ``[n] Title — topic · difficulty · score`` line."""
    lines: list[str] = []
    for s in sources:
        title = s.get("title") or s.get("source") or "?"
        bits = [str(title)]
        if s.get("topic"):
            bits.append(str(s["topic"]))
        if s.get("difficulty"):
            bits.append(str(s["difficulty"]))
        score = s.get("score")
        tail = f" (score {score})" if score is not None else ""
        lines.append(f"[{s.get('ref', '?')}] {' · '.join(bits)}{tail}")
    return lines


def to_records(history: list[dict[str, Any]]) -> list[TurnRecord]:
    """Normalise raw history messages into flat :class:`TurnRecord` rows."""
    records: list[TurnRecord] = []
    for i, msg in enumerate(history, start=1):
        meta = msg.get("meta") or {}
        records.append(
            TurnRecord(
                index=i,
                role=msg.get("role", "?"),
                content=str(msg.get("content", "")),
                model=meta.get("model"),
                level=meta.get("level"),
                tokens_in=meta.get("tokens_in"),
                tokens_out=meta.get("tokens_out"),
                cost=meta.get("cost"),
                sources=_source_lines(msg.get("sources") or []),
            )
        )
    return records


def conversation_to_json(history: list[dict[str, Any]]) -> str:
    """Full-fidelity JSON dump of the raw history (nothing dropped)."""
    return json.dumps(history, indent=2, default=str)


def conversation_to_csv(history: list[dict[str, Any]]) -> str:
    """One row per turn with flattened metadata; sources joined with ``|``."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "turn",
            "role",
            "content",
            "model",
            "level",
            "tokens_in",
            "tokens_out",
            "cost_usd",
            "sources",
        ]
    )
    for r in to_records(history):
        # Free-text cells (content, model, level, sources) pass through _csv_safe; numeric
        # cells can't start a formula so they are written as-is.
        writer.writerow(
            [
                r.index,
                _csv_safe(r.role),
                _csv_safe(r.content),
                _csv_safe(r.model or ""),
                _csv_safe(r.level or ""),
                r.tokens_in if r.tokens_in is not None else "",
                r.tokens_out if r.tokens_out is not None else "",
                f"{r.cost:.6f}" if r.cost is not None else "",
                _csv_safe(" | ".join(r.sources)),
            ]
        )
    return buf.getvalue()


def _pdf_safe(text: str) -> str:
    """Map common Unicode glyphs to ASCII and drop anything Latin-1 can't represent."""
    for bad, good in _GLYPHS.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "ignore").decode("latin-1")


def conversation_to_pdf(history: list[dict[str, Any]], title: str = "Synapse") -> bytes:
    """Render the conversation as a styled PDF transcript and return its bytes.

    Raises:
        RuntimeError: if ``fpdf2`` is not installed (add it to ``pyproject.toml`` and run
            ``make sync``). The UI catches this and shows an actionable message.
    """
    try:
        from fpdf import FPDF
    except ModuleNotFoundError as exc:  # dependency not synced yet
        raise RuntimeError(
            "PDF export needs the 'fpdf2' package — add it to pyproject.toml and run `make sync`."
        ) from exc

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, _pdf_safe(f"{title} — Conversation Transcript"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120)
    pdf.cell(0, 6, "AI/ML learning assistant", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(4)

    for r in to_records(history):
        heading = "You" if r.role == "user" else "Assistant"
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 70, 160) if r.role == "user" else pdf.set_text_color(20, 120, 80)
        pdf.cell(0, 8, _pdf_safe(f"{r.index}. {heading}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0)

        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _pdf_safe(r.content) or " ")

        if r.sources:
            pdf.ln(1)
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(90)
            pdf.multi_cell(0, 5, _pdf_safe("Sources:\n" + "\n".join(r.sources)))
            pdf.set_text_color(0)

        if r.model:
            usage = ""
            if r.tokens_in is not None and r.tokens_out is not None:
                usage = f" · {r.tokens_in}+{r.tokens_out} tokens"
            if r.cost is not None:
                usage += f" · ~${r.cost:.5f}"
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(140)
            pdf.cell(
                0,
                5,
                _pdf_safe(f"{r.model} · {r.level or ''} level{usage}"),
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.set_text_color(0)
        pdf.ln(4)

    out = pdf.output()  # fpdf2 returns a bytearray
    return bytes(out)
