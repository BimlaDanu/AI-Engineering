"""The **Past chats** sidebar panel, the New chat button, and the autosave that feeds them.

Three things that only make sense together: something has to write the conversation down
(:func:`autosave`), something has to let go of it without losing it (:func:`new_chat_button`),
and something has to list what was kept and let the reader destroy it
(:func:`past_chats_panel`).

Deleting is the part worth being careful about. Every other control in Synapse is reversible
by moving it back; these two rewrite the filesystem, and their reward for a misclick is losing
the thing the reader opened the panel to find. So both take **two presses**, and the pending
state is keyed per thread rather than held as one shared flag — a shared flag would, on the
rerun, arm the row immediately below the one just pressed, which is the row a reader is most
likely to press next.
"""

from __future__ import annotations

import uuid

import streamlit as st

from src import auth
from src.core.chats import ChatMeta, ChatStore, namespace_for, new_thread_id

THREAD_KEY = "chat_thread_id"
_ANON_KEY = "_anon_chat_id"
_PENDING_DELETE = "_chat_delete_armed"
_PENDING_DELETE_ALL = "_chat_delete_all_armed"

PAST_CHATS_SHOWN = 12
"""How many threads the panel draws at once.

The list is capped, the store is not. Every row is two widgets rebuilt on every rerun of
every page, so an unbounded list makes the whole sidebar feel like it is ignoring clicks once
a few dozen conversations have accumulated. **Delete all** still reaches the ones below the
cut, and the panel says so — a control that silently spared what it did not draw would be
worse than one that draws less.
"""


def _identity() -> str:
    """Name the current visitor for :func:`~src.core.chats.namespace_for`.

    Four cases, and the interesting one is the last. A signed-in visitor is their email; a
    password-gate user is their username. When no identity provider is configured at all
    there is exactly one person using this process — someone running ``make run`` on their own
    machine — so they get a stable namespace and their conversations are still there tomorrow.

    But on a deployment that *does* have a provider, an anonymous visitor is a stranger among
    other strangers, and a stable shared namespace would hand each of them the others'
    questions. They get a per-session id instead: their threads last as long as the session,
    which is the most that can be offered to someone who has not said who they are.
    """
    who = auth.visitor()
    if who.level == "signed_in":
        return who.email or who.name or "signed-in"
    if who.level == "password":
        user = auth.current_user()
        if user is not None:
            return f"password:{user.username}"
    if not auth.oidc_configured():
        return "local-operator"
    if _ANON_KEY not in st.session_state:
        st.session_state[_ANON_KEY] = uuid.uuid4().hex
    return f"anon:{st.session_state[_ANON_KEY]}"


def store() -> ChatStore:
    """This visitor's conversation store."""
    return ChatStore(namespace_for(_identity()))


def current_thread_id() -> str:
    """The thread the live conversation belongs to, minting one on first use."""
    if not st.session_state.get(THREAD_KEY):
        st.session_state[THREAD_KEY] = new_thread_id()
    return str(st.session_state[THREAD_KEY])


def autosave() -> None:
    """Write the live conversation to its thread, if there is anything to write.

    Called once per run from the entry point rather than from the chat page, because a
    reader who asks a question and then walks over to 📊 Evaluation has still had that
    conversation and should still find it in the list.

    Saving is best-effort all the way down (:meth:`~src.core.chats.ChatStore.save` swallows
    its own errors): a host with a read-only filesystem should lose the history feature, not
    the ability to answer questions.
    """
    history = st.session_state.get("history") or []
    if not history:
        return
    store().save(current_thread_id(), history)


def new_chat_button() -> None:
    """Start a fresh conversation, keeping the one it replaces.

    Nothing is deleted. The current thread is written out first and then let go of, so the
    conversation just cleared off the screen is the top row of **Past chats** — which is
    where somebody who has just pressed this by accident will go looking for it.
    """
    if not st.button("🆕 New chat", key="top_new_chat", help="Keeps the current one in Past chats"):
        return
    autosave()
    for key in ("history", "trace", "pending", THREAD_KEY):
        st.session_state.pop(key, None)
    st.session_state["totals"] = {"input": 0, "output": 0, "cost": 0.0}
    st.rerun()


def past_chats_panel() -> None:
    """List this visitor's saved conversations, and offer to open or delete them."""
    chats = store()
    threads = chats.list()
    with st.expander(f"🕘 Past chats ({len(threads)})", expanded=False):
        if not chats.available:
            st.caption(
                "Nothing is being kept — this host's filesystem is not writable, so a "
                "conversation lasts as long as the tab."
            )
            return
        if not threads:
            st.caption(
                "None yet. Starting a new chat will not delete the current one — it is kept "
                "here, and only stops being the conversation on screen."
            )
            return
        shown = threads[:PAST_CHATS_SHOWN]
        st.caption(
            "Newest first. Opening one replaces what is on screen; deleting one removes it "
            "for good."
        )
        if len(threads) > len(shown):
            st.caption(
                f"Showing the {len(shown)} most recent of {len(threads)}. Nothing has been "
                "deleted — the older ones are still stored, and **Delete all** reaches them too."
            )
        active = st.session_state.get(THREAD_KEY)
        for meta in shown:
            open_column, delete_column = st.columns([5, 1], vertical_alignment="center")
            if open_column.button(
                meta.label(),
                key=f"open_chat_{meta.id}",
                width="stretch",
                disabled=meta.id == active,
                help=f"{meta.turns} messages",
            ):
                _open(meta.id)
            with delete_column:
                _delete_one_button(meta)
        st.divider()
        _delete_all_button(len(threads))


def _open(thread_id: str) -> None:
    """Load a stored thread onto the screen, saving whatever it displaces."""
    autosave()
    chats = store()
    thread = chats.load(thread_id)
    if thread is None:
        st.warning("That conversation could not be read — it may already have been deleted.")
        return
    st.session_state["history"] = thread.messages
    st.session_state[THREAD_KEY] = thread.id
    st.session_state["trace"] = None
    st.rerun()


def _delete_one_button(meta: ChatMeta) -> None:
    """Offer to delete one conversation, in two presses.

    The armed set is keyed by thread id, so arming this row leaves its neighbours alone.

    The failure notice is a toast, not an ``st.warning``. Everything drawn before
    :func:`streamlit.rerun` is discarded, and the rerun below is not optional — the row has
    to be redrawn either way — so a warning here was written and then thrown away, leaving a
    failed delete looking exactly like a successful one. Toasts are queued across the rerun.
    """
    armed: set[str] = st.session_state.setdefault(_PENDING_DELETE, set())
    if meta.id in armed:
        if st.button("✓", key=f"confirm_del_{meta.id}", help="Delete for good", type="primary"):
            armed.discard(meta.id)
            if store().delete(meta.id):
                if st.session_state.get(THREAD_KEY) == meta.id:
                    # The thread just deleted is the one on screen. Clear it rather than
                    # leaving a conversation displayed that no longer exists anywhere.
                    st.session_state.pop("history", None)
                    st.session_state.pop(THREAD_KEY, None)
            else:
                st.toast("That conversation could not be deleted — it is still listed.", icon="⚠️")
            st.rerun()
        return
    if st.button("🗑", key=f"del_{meta.id}", help="Delete this conversation"):
        armed.add(meta.id)
        st.rerun()


def _delete_all_button(total: int) -> None:
    """Offer to clear the whole store at once, in two presses.

    Below the rows rather than above them, so a control that empties the list does not sit
    where a reader aims for the newest conversation in it. The count is every thread stored,
    not the ones drawn, so the number on the button is the number that will actually go.
    """
    if st.session_state.get(_PENDING_DELETE_ALL):
        left, right = st.columns(2)
        if left.button("Delete all", key="confirm_del_all", type="primary", width="stretch"):
            removed = store().delete_all()
            st.session_state[_PENDING_DELETE_ALL] = False
            st.session_state.pop("history", None)
            st.session_state.pop(THREAD_KEY, None)
            st.toast(f"Deleted {removed} conversation{'s' if removed != 1 else ''}.")
            st.rerun()
        if right.button("Cancel", key="cancel_del_all", width="stretch"):
            st.session_state[_PENDING_DELETE_ALL] = False
            st.rerun()
        st.caption(f"This removes all {total} for good and cannot be undone.")
        return
    if st.button(f"🗑 Delete all {total} chats", key="del_all", width="stretch"):
        st.session_state[_PENDING_DELETE_ALL] = True
        st.rerun()
