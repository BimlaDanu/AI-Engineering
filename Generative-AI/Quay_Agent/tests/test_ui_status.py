"""A monitor that lies is worse than no monitor, so its facts are tested.

`src/ui/status.py` is what every page on the web monitor reports from. The risk it
carries is specific: a status page keeps rendering long after the thing it watches has
broken, because nothing ever checks the checker. These tests pin the two claims that
would be most damaging to get wrong -- that a module is present when it is not, and that
a dependency is installed when it is not.

Streamlit is never imported here. The pages are a thin rendering of this module, which
is exactly why the split exists.
"""

from __future__ import annotations

from src.ui import status


def test_the_repository_root_is_found_from_the_file_not_the_cwd() -> None:
    # A monitor launched from the wrong directory must not report an empty project.
    assert (status.PROJECT_ROOT / "pyproject.toml").is_file()
    assert (status.PROJECT_ROOT / "src").is_dir()


def test_component_presence_is_read_from_disk() -> None:
    by_path = {component.path: component for component in status.COMPONENTS}

    written = by_path["src/physics/quantum/hamiltonians.py"]
    assert written.exists
    assert written.lines > 100, "a real module, not an empty placeholder"

    # The absent case is built here rather than taken from COMPONENTS, which used to
    # supply one. Reaching into the list for a module that does not exist makes the
    # test pass only while the project is unfinished, and it went on passing after
    # the work was done because seven entries still named files that had been renamed.
    unwritten = status.Component("src/agent/nothing_here.py", 1, "a path that was never written")
    assert not unwritten.exists
    assert unwritten.lines == 0


def test_every_component_the_page_lists_is_a_file_that_exists() -> None:
    """The progress figure counts real modules, so a rename cannot deflate it.

    ``build_progress`` drives a completeness number on a page a reviewer reads. Seven
    of the fifty-five entries named modules that had been renamed to satisfy the house
    naming rule -- ``gradients.py`` folded into ``statevector.py``, ``varqite.py``
    became ``imaginary_time_evolution.py`` -- so a finished project reported itself as
    48 of 55. Under-reporting is still misreporting.
    """
    missing = [component.path for component in status.COMPONENTS if not component.exists]

    assert not missing, f"listed on the status page but absent from the tree: {missing}"


def test_build_progress_counts_only_what_exists() -> None:
    built, total = status.build_progress()
    assert total == len(status.COMPONENTS)
    assert built == sum(1 for component in status.COMPONENTS if component.exists)
    # Not `built < total`, which this asserted while the project was being written and
    # then quietly became a test that the project stays unfinished. Everything listed
    # is now built, and the invariant worth holding is that the two agree.
    assert 0 < built <= total


def test_every_component_sits_in_exactly_one_tier() -> None:
    tiers = {component.tier for component in status.COMPONENTS}
    assert tiers, "the checklist cannot be empty"
    assert min(tiers) >= 1
    assert sum(len(status.components_in_tier(tier)) for tier in tiers) == len(status.COMPONENTS)


def test_installed_and_absent_dependencies_are_told_apart() -> None:
    by_name = {dependency.distribution: dependency for dependency in status.DEPENDENCIES}
    # Declared in pyproject and therefore synced into the environment.
    assert by_name["numpy"].installed_version is not None
    assert by_name["streamlit"].installed_version is not None


def test_missing_dependencies_are_filtered_by_the_tier_that_needs_them() -> None:
    # Nothing is ever missing "before it is needed": at the first tier a package only
    # a later one requires is not a problem, and a monitor that says otherwise teaches
    # the reader to ignore it.
    first = status.missing_dependencies(1)
    everything = status.missing_dependencies(99)
    assert set(first) <= set(everything)
    assert all(dependency.needed_from_tier <= 1 for dependency in first)
    assert all(dependency.installed_version is None for dependency in everything)


def test_artefacts_are_listed_newest_first_and_stay_inside_the_project() -> None:
    found = status.artefacts()
    timestamps = [artefact.modified for artefact in found]
    assert timestamps == sorted(timestamps, reverse=True)
    for artefact in found:
        assert not artefact.path.startswith(("/", ".."))


def test_corpus_size_counts_every_shelf_not_just_the_root() -> None:
    # The notes live one directory down, one directory per shelf. A root-only
    # glob returns zero and the monitor then reports an empty corpus while the
    # retriever is happily indexing nineteen notes.
    corpus = status.PROJECT_ROOT / "data" / "corpus"
    assert status.corpus_size() == len(list(corpus.rglob("*.md")))
    assert status.corpus_size() > 0, "the committed corpus is not empty"


def test_every_watched_directory_has_a_description_and_no_others_do() -> None:
    """The About page's list of watched directories is the list it actually watches.

    The page used to print its own hand-written list, and it drifted: it named
    ``reports/circuits/`` and ``data/cases/``, neither of which is in
    ``ARTEFACT_DIRECTORIES``, and left out ``reports/submissions``, which is. A
    reader was told the software watches two directories it never looks at. The
    page now renders this mapping, so the only way to drift is to fail here.
    """
    described = set(status.ARTEFACT_PURPOSE)
    watched = set(status.ARTEFACT_DIRECTORIES)

    assert described == watched, (
        f"described but not watched: {sorted(described - watched)}; "
        f"watched but not described: {sorted(watched - described)}"
    )
