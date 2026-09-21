r"""Every configuration priced against every machine, as a table that is committed.

The rest of the evaluation harness runs campaigns and grades what they conclude.
This does something narrower and more useful as a foundation: it works out what
each configuration *costs* on each machine, before any agent has an opinion about
it. The output is the resource oracle -- the table a claim in a report can be
checked against, and one that can be read without running anything.

Why it can be committed
-----------------------

Nothing here is a measurement. Every figure is arithmetic over a machine's wiring
diagram and its published rates: place the spins, walk the shortest paths, count
the layers, multiply the error probabilities. There is no model, no credential, no
random number and no simulation, so the same command produces byte-identical output
on any machine on any day. A table with that property belongs in the repository
rather than in a cache -- it is documentation that happens to be generated.

The one thing it is not
-----------------------

It says nothing about whether an answer would be *correct*. A configuration can be
perfectly affordable and still represent the ground state badly, and one that is
unaffordable might have been excellent. Cost and accuracy are separate axes and
conflating them is the standard way a feasibility study goes wrong: this file is
the cost axis alone, and :mod:`src.evals.harness` is where accuracy is measured.

Resuming
--------

Rows already in the file are kept and only missing ones are computed. That is not
for speed -- the whole sweep takes a second -- but so that a row can be added to the
grid without silently rewriting the ones that were already reviewed. ``--rebuild``
recomputes everything, which is what to run after changing a device's figures.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.figure_export import PROJECT_ROOT
from src.hardware.devices import DEVICES, Device
from src.hardware.fidelity import depth_ceiling, estimate
from src.hardware.transpile import fits, transpile
from src.logging_setup import configure_logging, get_logger
from src.physics.quantum.ansatz import AnsatzSpec

SWEEP_PATH = PROJECT_ROOT / "data" / "resource-sweep.json"
"""Where the table is written.

Under ``data`` rather than ``reports`` because it is an input to other things --
a figure, a claim in a report, a check in a review -- rather than a document
somebody reads end to end.
"""

CHAIN_LENGTHS: tuple[int, ...] = (4, 6, 8, 10, 12, 16)
"""Chain lengths swept.

Twelve is where the project works and sixteen is the ceiling every method here
enforces (:data:`~src.physics.model.MAX_SITES_SPARSE`), so a sweep that ran past
sixteen would be tabulating rows no method would answer. The twenty this used to
end on was chosen from the heavy-hexagon lattice's longest unbroken run of qubits,
which is a fact about a device rather than a size worth paying twelve seconds for.
"""

CIRCUIT_DEPTHS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12)
"""Circuit depths swept. Dense where the ceiling falls, sparse past it."""

BOUNDARIES: tuple[Literal["open", "periodic"], ...] = ("open", "periodic")
"""Both boundary conditions, because the difference between them is the finding."""

EXIT_OK = 0
"""Exit status when the table was written."""

_logger = get_logger("evals.sweep")


@dataclass(frozen=True, slots=True)
class Configuration:
    """One cell of the grid: a problem, a shape and a machine.

    Attributes:
        n_sites: Length of the chain.
        boundary: ``"open"`` for a segment, ``"periodic"`` for a ring.
        depth: Number of ansatz layers.
        device: Name of the machine.
    """

    n_sites: int
    boundary: Literal["open", "periodic"]
    depth: int
    device: str

    @property
    def key(self) -> str:
        """A stable identifier, so a row can be found again without a database.

        Returns:
            The four fields joined by slashes.
        """
        return f"{self.device}/{self.n_sites}/{self.boundary}/p{self.depth}"


def grid() -> Iterator[tuple[Configuration, Device]]:
    """Enumerate every configuration worth pricing.

    Configurations that do not fit on a machine are skipped rather than recorded as
    failures: a chain longer than the register is not a result about that chain, and
    a table full of them would hide the rows that say something.

    Yields:
        Each configuration alongside the machine it belongs to.
    """
    for device in DEVICES:
        for n_sites in CHAIN_LENGTHS:
            if fits(n_sites, device) is not None:
                continue
            for boundary in BOUNDARIES:
                for depth in CIRCUIT_DEPTHS:
                    yield (
                        Configuration(
                            n_sites=n_sites,
                            boundary=boundary,
                            depth=depth,
                            device=device.name,
                        ),
                        device,
                    )


def price(configuration: Configuration, device: Device) -> dict[str, Any]:
    """Work out what one configuration costs on one machine.

    Args:
        configuration: The cell of the grid.
        device: The machine it names.

    Returns:
        A flat row of primitives: what the wiring cost, how long the circuit runs,
        what survives, and whether the depth is inside the machine's ceiling.
    """
    compiled = transpile(
        AnsatzSpec(
            n_qubits=configuration.n_sites,
            depth=configuration.depth,
            boundary=configuration.boundary,
        ),
        device,
    )
    surviving = estimate(compiled)
    ceiling = depth_ceiling(configuration.n_sites, device, boundary=configuration.boundary)
    inflation = surviving.shot_inflation()
    return {
        "key": configuration.key,
        "device": configuration.device,
        "n_sites": configuration.n_sites,
        "boundary": configuration.boundary,
        "depth": configuration.depth,
        "swaps": compiled.swaps,
        "two_qubit_depth": compiled.two_qubit_depth,
        "abstract_two_qubit_depth": compiled.ansatz.two_qubit_depth,
        "routing_overhead": round(compiled.routing_overhead, 3),
        "duration_us": round(compiled.duration_us, 3),
        "fraction_of_t2": round(compiled.duration_ns / device.t2_ns, 4),
        "fidelity": round(surviving.total, 5),
        "dominant_loss": surviving.dominant_loss,
        "shot_inflation": None if inflation == float("inf") else round(inflation, 2),
        "max_depth": ceiling.limit,
        "binding_constraint": ceiling.binding,
        "within_ceiling": configuration.depth <= ceiling.limit,
    }


def existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Read whatever the table already holds, keyed for lookup.

    A malformed or absent file is treated as empty rather than as an error. The
    sweep is cheap enough to redo, and refusing to run because a previous run was
    interrupted mid-write would be the wrong trade.

    Args:
        path: Where the table lives.

    Returns:
        The rows already computed, keyed by :attr:`Configuration.key`.
    """
    if not path.exists():
        return {}
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _logger.warning("sweep_unreadable", extra={"path": str(path)})
        return {}
    rows = stored.get("rows", []) if isinstance(stored, dict) else []
    return {row["key"]: row for row in rows if isinstance(row, dict) and "key" in row}


def run(path: Path = SWEEP_PATH, rebuild: bool = False) -> tuple[int, int]:
    """Fill in the table, keeping whatever is already there.

    Args:
        path: Where to write.
        rebuild: Recompute every row rather than only the missing ones.

    Returns:
        How many rows were computed this time, and how many the table now holds.
    """
    kept = {} if rebuild else existing_rows(path)
    rows: list[dict[str, Any]] = []
    computed = 0
    for configuration, device in grid():
        if configuration.key in kept:
            rows.append(kept[configuration.key])
            continue
        rows.append(price(configuration, device))
        computed += 1
    payload = {
        "what": "resource cost of each circuit configuration on each machine",
        "how": (
            "arithmetic over each machine's coupling graph and published rates. "
            "No simulation, no model, no random numbers -- the same command "
            "reproduces this file exactly."
        ),
        "devices": [device.describe() for device in DEVICES],
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _logger.info("sweep_written", extra={"computed": computed, "rows": len(rows)})
    return computed, len(rows)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command line.

    Args:
        argv: Arguments to parse. ``None`` reads the real command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.evals.sweep",
        description="Price every circuit configuration against every machine.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="recompute every row rather than only the missing ones",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=SWEEP_PATH,
        help="where to write the table",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep from a terminal.

    Args:
        argv: Command-line arguments. ``None`` reads the real ones.

    Returns:
        The process exit status.
    """
    configure_logging()
    arguments = parse_arguments(argv)
    computed, total = run(path=arguments.out, rebuild=arguments.rebuild)
    print(f"{total} configurations priced ({computed} computed now) -> {arguments.out}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
