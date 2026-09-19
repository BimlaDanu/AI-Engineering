"""Offline tests for the saved-conversation store (:mod:`src.core.chats`).

The store is framework-agnostic by design — it takes a root path and a namespace and imports
no Streamlit — so the whole of it is exercised here against ``tmp_path`` with no browser.

Three properties carry most of the weight:

* **Deleting actually deletes.** A delete button that reports success without removing the
  file is worse than no delete button, because the reader stops looking for the conversation.
* **One visitor's Delete all cannot reach another's threads.** On a deployment everyone
  shares one process and one filesystem; the namespace is the only thing between them.
* **A thread id from a widget key never becomes an arbitrary path.** Ids are validated, not
  sanitised — a "cleaned" id would silently address the wrong file.
"""

from __future__ import annotations

import json

from src.core.chats import (
    ChatStore,
    namespace_for,
    new_thread_id,
    title_from,
)

_CONVERSATION = [
    {"role": "user", "content": "What is a transformer?"},
    {"role": "assistant", "content": "A sequence model.", "sources": [{"ref": 1}]},
]


def _store(tmp_path, who: str = "alice") -> ChatStore:
    return ChatStore(namespace_for(who), root=tmp_path)


# --- Saving and listing ---------------------------------------------------------------------


def test_saving_then_listing_returns_the_thread(tmp_path):
    store = _store(tmp_path)
    thread_id = new_thread_id()
    store.save(thread_id, _CONVERSATION)

    rows = store.list()
    assert [row.id for row in rows] == [thread_id]
    assert rows[0].title == "What is a transformer?"
    assert rows[0].turns == 2


def test_a_reopened_thread_keeps_its_citations(tmp_path):
    # The whole message is stored, not just role/content. A transcript that dropped its
    # sources would undercut the one promise the project makes about its answers.
    store = _store(tmp_path)
    thread_id = new_thread_id()
    store.save(thread_id, _CONVERSATION)

    loaded = store.load(thread_id)
    assert loaded is not None
    assert loaded.messages[1]["sources"] == [{"ref": 1}]


def test_a_message_that_will_not_serialise_degrades_to_its_text(tmp_path):
    # A page is free to hang anything off a message; the conversation must still save.
    store = _store(tmp_path)
    thread_id = new_thread_id()
    store.save(thread_id, [{"role": "user", "content": "hi", "obj": object()}])

    loaded = store.load(thread_id)
    assert loaded is not None
    assert loaded.messages == [{"role": "user", "content": "hi"}]


def test_an_empty_conversation_is_not_saved(tmp_path):
    # An empty thread in the list is a row that does nothing but invite a delete.
    store = _store(tmp_path)
    assert store.save(new_thread_id(), []) is None
    assert store.list() == []


def test_saving_again_replaces_rather_than_duplicates(tmp_path):
    store = _store(tmp_path)
    thread_id = new_thread_id()
    store.save(thread_id, _CONVERSATION)
    store.save(thread_id, [*_CONVERSATION, {"role": "user", "content": "and attention?"}])

    rows = store.list()
    assert len(rows) == 1
    assert rows[0].turns == 3


def test_listing_is_newest_first(tmp_path):
    store = _store(tmp_path)
    first, second = new_thread_id(), new_thread_id()
    store.save(first, [{"role": "user", "content": "older"}])
    store.save(second, [{"role": "user", "content": "newer"}])

    # `updated` is written at save time, so the second save sorts ahead of the first.
    assert next(row.title for row in store.list()) == "newer"


def test_an_unreadable_file_is_skipped_not_fatal(tmp_path):
    # One corrupt thread must not take the list — including its delete buttons — down with it.
    store = _store(tmp_path)
    good = new_thread_id()
    store.save(good, _CONVERSATION)
    (store.directory / f"{new_thread_id()}.json").write_text("{not json", encoding="utf-8")

    assert [row.id for row in store.list()] == [good]


def test_a_saved_thread_is_readable_json(tmp_path):
    # Not an implementation detail: a reader should be able to open their own transcript
    # without this application, which is the point of a file per conversation.
    store = _store(tmp_path)
    thread_id = new_thread_id()
    store.save(thread_id, _CONVERSATION)

    payload = json.loads((store.directory / f"{thread_id}.json").read_text(encoding="utf-8"))
    assert payload["messages"][0]["content"] == "What is a transformer?"
    assert payload["created"] and payload["updated"]


# --- Deleting -------------------------------------------------------------------------------


def test_deleting_one_removes_only_that_thread(tmp_path):
    store = _store(tmp_path)
    doomed, kept = new_thread_id(), new_thread_id()
    store.save(doomed, [{"role": "user", "content": "delete me"}])
    store.save(kept, [{"role": "user", "content": "keep me"}])

    assert store.delete(doomed) is True
    assert [row.id for row in store.list()] == [kept]


def test_deleting_reports_false_when_nothing_was_removed(tmp_path):
    # The caller uses this to tell the reader it is still there, rather than claiming a
    # deletion that did not happen.
    store = _store(tmp_path)
    assert store.delete(new_thread_id()) is False


def test_delete_all_clears_the_store_and_counts_what_went(tmp_path):
    store = _store(tmp_path)
    for index in range(3):
        store.save(new_thread_id(), [{"role": "user", "content": f"q{index}"}])

    assert store.delete_all() == 3
    assert store.list() == []


def test_delete_all_on_an_empty_store_is_harmless(tmp_path):
    assert _store(tmp_path).delete_all() == 0


def test_one_visitors_delete_all_cannot_reach_anothers_threads(tmp_path):
    # The privacy property the namespace exists for. On a deployment every visitor shares one
    # filesystem, and this is the only thing keeping their conversations apart.
    alice, bob = _store(tmp_path, "alice@example.com"), _store(tmp_path, "bob@example.com")
    alice.save(new_thread_id(), [{"role": "user", "content": "alice's question"}])
    bob.save(new_thread_id(), [{"role": "user", "content": "bob's question"}])

    assert alice.delete_all() == 1
    assert len(bob.list()) == 1


def test_one_visitor_cannot_list_anothers_threads(tmp_path):
    alice, bob = _store(tmp_path, "alice@example.com"), _store(tmp_path, "bob@example.com")
    alice.save(new_thread_id(), [{"role": "user", "content": "private"}])

    assert bob.list() == []


# --- Ids are validated, not sanitised --------------------------------------------------------


def test_a_traversing_id_is_refused_rather_than_cleaned(tmp_path):
    store = _store(tmp_path)
    outside = tmp_path / "secret.json"
    outside.write_text("{}", encoding="utf-8")

    assert store.load("../../secret") is None
    assert store.delete("../../secret") is False
    assert store.save("../../secret", _CONVERSATION) is None
    assert outside.exists(), "a refused id must not have reached the filesystem"


def test_a_non_uuid_id_is_refused(tmp_path):
    store = _store(tmp_path)
    assert store.save("not-a-uuid", _CONVERSATION) is None
    assert store.list() == []


# --- Naming ---------------------------------------------------------------------------------


def test_the_title_is_the_opening_question(tmp_path):
    assert title_from([{"role": "user", "content": "  Why   attention?  "}]) == "Why attention?"


def test_a_long_question_is_truncated_with_an_ellipsis():
    title = title_from([{"role": "user", "content": "word " * 50}])
    assert len(title) <= 60
    assert title.endswith("…")


def test_a_conversation_with_no_question_still_gets_a_label():
    # A row with an empty label is a row a reader cannot aim at, including to delete it.
    assert title_from([{"role": "assistant", "content": "unprompted"}]) == "Untitled conversation"


def test_the_namespace_does_not_reveal_the_identity():
    # A server-side directory listing should not become a list of who has used the app.
    digest = namespace_for("alice@example.com")
    assert "alice" not in digest
    assert digest == namespace_for("alice@example.com"), "namespaces must be stable"
    assert digest != namespace_for("bob@example.com")
