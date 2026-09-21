"""Function calling: letting the model ask for work, without letting it do any.

This is the one place a language model decides *what happens* rather than what to
say, so the split is worth stating precisely.

**The model chooses the call. This module makes it.** A tool call arrives as a
name and a bag of JSON arguments -- a request, not an action. The name is looked
up in :class:`Toolbox`, the arguments are validated against a schema, and only
then does tested Python run. A name that does not exist and arguments of the wrong
shape both come back as a recorded refusal, which is why a confused model costs a
sentence rather than a traceback.

**One round, not a loop.** The usual pattern feeds the results back and lets the
model call again until it stops. That is not done here: each of these tools
answers independently, so a second round buys nothing, while a loop is where the
cost, the latency and the injection surface all live -- every result fed back is
another chance for retrieved text to steer the next call. One round is a ceiling
that cannot be argued with.

**The results are records, not conversation.** A run returns
:class:`ToolRun` values which the graph puts in its state. Nothing here writes
into a message history the model then reads as its own reasoning.

**What the tools can and cannot do** follows from where they live. Solving a
comparison chain goes through the same cross-check as the main run, so it cannot
produce an unverified number; the arXiv lookup cannot produce a verified one, and
is labelled accordingly wherever it surfaces. Neither can widen a cost gate: see
:data:`~src.tools.chains.MAX_COMPARISON_SITES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from src.agent.llm import as_data, chat_model_or_none
from src.logging_setup import get_logger, log_llm_call
from src.physics.model import TFIMSpec
from src.tools import mcp, websearch
from src.tools.chains import MAX_COMPARISON_SITES, MIN_COMPARISON_SITES, ChainRun, compare_chain
from src.tools.http import Fetcher
from src.tools.mcp import McpCall, McpServer, Transport
from src.tools.papers import MAX_PAPERS, Fetch, PaperSearch, find_papers
from src.tools.sweeps import (
    DEFAULT_POINTS,
    MAX_COSTLY_SITES,
    FieldSweep,
    Observable,
    sweep_field,
)
from src.tools.websearch import MAX_RESULTS as MAX_WEB_RESULTS
from src.tools.websearch import Search, WebSearch, search_web
from src.tools.wiki import WikiLookup, look_up

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from langchain_core.language_models import BaseChatModel

    from src.settings import Settings

LOG = get_logger("tools.calling")

MAX_TOOL_CALLS = 3
"""Most tool calls executed from one model reply.

A model may ask for as many as it likes; this many are run. The ceiling exists
because model-chosen fan-out is a cost the user never authorised -- a reply asking
for twelve comparison chains is one diagonalisation short of a hung page. Calls
beyond the ceiling are dropped and reported, never silently.
"""

CONSULT_SYSTEM = """You decide whether a physics answer that is already prepared \
would be improved by one or two extra pieces of work.

Answer with tool calls and nothing else. Do not write prose, do not explain your \
choice, and do not answer the question yourself -- another step does that, and any \
text you write here is discarded unread. If no tool is needed, return an empty \
reply. Writing a paragraph instead of calling a tool is the one wrong answer here: \
it costs the user twenty seconds and a thousand tokens and changes nothing.

You are shown the question and everything that has been established so far: the \
chain that was solved, the numbers that were computed and cross-checked, and any \
passages retrieved from the project's literature notes.

You are asked at most once per question, and only once the knowledge base has been \
searched to its limit and what came back still did not cover the question. The \
default is therefore not "nothing is needed": something is missing, and this is the \
last step that can go and find it. Call a tool when it would change the answer, and \
call none only when nothing offered below could.

Ask for a comparison chain when the question is about how a quantity behaves as \
the chain grows, or how close a finite result is to the infinite-chain limit. One \
or two extra lengths is enough to show a trend.

Ask for a field sweep when the question is about a curve rather than a value: how \
the magnetisation turns on, how the energy varies with the field, how the low-lying \
levels move, where the chain looks critical, or anything the user asks to see \
plotted. Set its observable, and be careful which one you pick: the magnetisation \
and the ground-state energy are properties **of the ground state**, while the \
low-lying energy spectrum, the levels, the excitations and the gap are about the \
states **above** it, and the derivatives of the energy with respect to the field are \
a third thing again. A question about excitations is not answered by a ground-state \
curve, a question about the ground state is not answered by a spectrum, and a \
question about the first and second derivatives of the energy is not answered by a \
magnetisation curve -- ask for 'derivatives' and you are given both derivatives, \
computed in closed form.

Ask for papers when the question is about what the literature says and the \
retrieved notes did not answer it -- including when no passages were retrieved at \
all, which is the case where an outside source is most useful. If the question \
names a person, put that name in the author field rather than in the keywords.

Ask Wikipedia for background the notes do not carry: what a term means outside \
this project, who someone was, what a technique is called in other fields.

Reach for the web, or for a remote tool, only when everything above has failed. \
Those are pointers, not evidence.

Outside sources are ordered by how much they are worth: the project's own notes, \
then arXiv, then Wikipedia, then anything else. Prefer the highest one that can \
answer the question, and never use any of them to support a number.

Never call a tool to check a number that was already cross-checked."""


class CompareChain(BaseModel):
    """Solve the same Hamiltonian at a different chain length, for comparison.

    Use this when the question is about finite-size behaviour: how a quantity
    changes as the chain grows, or how far a short chain is from the infinite-chain
    limit. The coupling, the field and the boundary condition are fixed by the
    user and cannot be changed here -- only the number of spins.

    Attributes:
        n_sites: How many spins the comparison chain should have.
    """

    model_config = ConfigDict(frozen=True)

    n_sites: int = Field(
        description=(
            f"Number of spins in the comparison chain, from {MIN_COMPARISON_SITES} "
            f"to {MAX_COMPARISON_SITES}. Longer chains need the user's approval and "
            "will be declined here."
        )
    )


class SweepField(BaseModel):
    """Trace a quantity across the whole field range and plot it.

    Use this when the question is about a curve rather than a single value: how
    the magnetisation turns on as the field grows, how the low-lying levels move as
    the field is turned up, where this chain looks critical, or anything the user
    asks to see plotted. Every point is computed by the physics layer, and the
    ground state at each one by both solvers independently where both apply, so the
    curve is verified the same way a single number is.

    **Say which quantity the question is about.** The ground-state curves and the
    excitation spectrum are different physics -- one is an expectation value in the
    ground state, the other is the energies of the states above it -- and asking for
    the wrong one answers a question nobody asked.

    Attributes:
        observable: Which curve the question wants.
        points: How many field values to sample, or ``None`` to sweep at the
            resolution the user chose.
    """

    model_config = ConfigDict(frozen=True)

    observable: Observable = Field(
        default="ground_state",
        description=(
            "Which quantity to plot. 'ground_state' for the transverse "
            "magnetisation and the ground-state energy density -- both are "
            "properties of the ground state. 'spectrum' for the low-lying "
            "excitation energies E_n - E_0, which is what a question about the "
            "energy spectrum, the levels, the excitations or the gap is asking "
            "for. 'derivatives' for the ground-state energy density together with "
            "its first and second derivatives with respect to the field, which is "
            "what a question naming derivatives, dE/dh, curvature or the second "
            "derivative is asking for -- pick this rather than 'ground_state' "
            "there, because a magnetisation curve does not answer it. 'both' only "
            "when the question genuinely asks for both."
        ),
    )
    points: int | None = Field(
        default=None,
        description=(
            "How many field values to sample between h = 0 and h = 2J. Leave this "
            "out and the sweep runs at the resolution the user chose, which is what "
            "nearly every question wants. Set it only to ask for a *coarser* curve "
            "than that -- a sketch of the shape, a handful of points. Asking for a "
            "finer one has no effect: the user's setting is the ceiling."
        ),
    )


class FindPapers(BaseModel):
    """Search arXiv by keyword, title or author.

    Use this only when the question asks what the literature says and the
    project's own notes did not answer it. Results are not verified and cannot be
    used as evidence for a number or a claim -- they are further reading.

    Fill in whichever fields the question gives you and leave the rest empty. An
    author's name goes in ``author`` rather than into ``query``: searched there it
    is matched against the author list in every spelling arXiv indexes, while the
    same name in ``query`` only finds papers that happen to mention it.

    Attributes:
        query: Keywords to search abstracts for.
        author: An author's name, in any order.
        title: Words that should appear in the title.
        limit: How many papers to return.
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        default="",
        description="A few keywords to search paper abstracts for. Not a sentence.",
    )
    author: str = Field(
        default="",
        description=(
            "One author's name, given name first or surname first -- both work, and "
            "a surname on its own works. Leave empty unless the question names a person."
        ),
    )
    title: str = Field(
        default="",
        description="Words that should appear in the paper's title. Leave empty if unsure.",
    )
    limit: int = Field(
        default=3,
        description=f"How many papers to return, at most {MAX_PAPERS}.",
    )


class LookUpWikipedia(BaseModel):
    """Look a term or a topic up in Wikipedia.

    Use this for background the project's own notes do not carry: what a term
    means in the wider literature, who someone was, what a technique is called
    elsewhere. It is general reference, not evidence -- it can never support a
    number, and every number in this project comes from the physics layer.

    Attributes:
        query: What to look up.
        deep: Whether to read the whole article rather than its opening.
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        description="The thing to look up -- a term or a title, not a whole question."
    )
    deep: bool = Field(
        default=False,
        description=(
            "False reads the opening definition, which is enough to explain a term. "
            "True reads the whole article and lists its sections -- ask for that when "
            "the question is about the subject itself rather than about one word in it."
        ),
    )


class SearchWeb(BaseModel):
    """Search the open web.

    The last resort, and the least trustworthy source here: an arbitrary page, no
    review of any kind, nothing in this application able to check it. Use it only
    when the notes, arXiv and Wikipedia have all failed, and treat what comes back
    as a pointer to follow rather than as something to state.

    Attributes:
        query: What to search for.
        limit: How many results to return.
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="What to search the web for.")
    limit: int = Field(
        default=3,
        description=f"How many results to return, at most {MAX_WEB_RESULTS}.",
    )


class CallMcpTool(BaseModel):
    """Call a tool on the connected MCP server.

    MCP is how this agent borrows tools it does not own, so what is available here
    depends entirely on which server is connected -- ask for its tool list if you
    do not know. Whatever comes back is a third party's output and is not verified.

    Attributes:
        tool: The remote tool's name, exactly as the server advertises it.
        query: The single text argument to pass it.
    """

    model_config = ConfigDict(frozen=True)

    tool: str = Field(
        description=(
            "The remote tool's name, exactly as advertised. A name the server does "
            "not offer is declined, and the reply lists what it does offer."
        )
    )
    query: str = Field(description="The text to pass as that tool's argument.")


@dataclass(frozen=True, slots=True)
class ToolRun:
    """One tool call and what came of it.

    Attributes:
        tool: The name the model asked for, as it asked for it. Kept verbatim
            even when no such tool exists, because the wrong name is the useful
            part of that record.
        arguments: The arguments as received, before validation.
        ok: Whether the call produced something usable.
        detail: One sentence on what happened. Populated either way.
        chain: The comparison run, when this was :class:`CompareChain`.
        sweep: The curve, when this was :class:`SweepField`.
        papers: The lookup, when this was :class:`FindPapers`.
        wiki: The article, when this was :class:`LookUpWikipedia`.
        web: The results, when this was :class:`SearchWeb`.
        remote: The remote output, when this was :class:`CallMcpTool`.
    """

    tool: str
    arguments: dict[str, Any]
    ok: bool
    detail: str
    chain: ChainRun | None = None
    sweep: FieldSweep | None = None
    papers: PaperSearch | None = None
    wiki: WikiLookup | None = None
    web: WebSearch | None = None
    remote: McpCall | None = None

    @property
    def verified(self) -> bool:
        """Whether this result went through the project's own cross-check.

        Returns:
            ``True`` only for the tools that reach inward and run the physics.
            Everything that reaches outside the process is unverifiable by
            construction, and one property saying so is better than five call
            sites remembering to.
        """
        return self.ok and (self.chain is not None or self.sweep is not None)

    def explain(self) -> str:
        """Say what this call did, in one line, for the machinery layer.

        Returns:
            The tool, its arguments and the outcome -- the tool's own sentence
            where it has one, so the user reads what the code decided rather than
            a paraphrase of it.
        """
        for result in (self.chain, self.sweep, self.papers, self.wiki, self.web, self.remote):
            if result is not None:
                return f"{self.tool}{self.arguments}: {result.explain()}"
        return f"{self.tool}{self.arguments}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Toolbox:
    """The tools available for one question, and the machinery to run them.

    Built per run, because two of the three things a tool needs -- the chain the
    question is about and whether the user allowed network calls -- are properties
    of that run rather than of the process.

    Attributes:
        spec: The chain the question is about. A comparison run and a sweep both
            inherit their Hamiltonian from here.
        use_comparison: Whether the comparison run at another chain length is
            offered. Switched off for a question that asked for no number, by
            :func:`src.agent.router.asks_for_a_number`.
        use_sweep: Whether the field sweep is offered. **Only a question that asked
            for a curve gets it**, which is a much narrower test than "asked for a
            number" and is the one this tool needs: a sweep *is* a curve, and there
            is no reading of *what is the gap at L = 8* that wants twenty-one points.

            The two are separate flags because gating them together was not enough.
            *How does the parity operator affect the boundary conditions?* mentions
            the ground state, so any test built on naming a quantity passes it, and
            the answer came back with a twenty-one point sweep, a figure and a
            cross-check summary attached to a question about algebra. Vocabulary
            cannot tell a quantity that is mentioned from one that is requested, so
            the sweep is gated on the shape of the answer wanted instead -- see
            :attr:`src.agent.router.Routing.wants_curve`, which is already the state
            the decider uses for exactly this distinction.

            Both follow the rule the flags below give: a tool the model cannot see is
            a tool it cannot be talked into wanting. It called the sweep because it
            was there.
        use_arxiv: Whether the arXiv lookup is offered at all. Off means the tool
            is not even described to the model, which is stronger than declining
            the call: a tool the model cannot see is a tool it cannot be talked
            into wanting. On by default, because the corpus is small and a
            question it does not cover is better met with a real paper than with a
            refusal.
        use_wikipedia: Whether the encyclopedia lookup is offered. On by default,
            for the same reason and at the same level of trust.
        use_web: Whether open web search is offered. Off by default -- it is the
            least trustworthy source here, and it costs money -- and it is
            withheld anyway unless the deployment configured an endpoint.
        use_mcp: Whether remote MCP tools are offered. Withheld unless a server is
            configured, since there is nothing to call otherwise.
        max_papers: Ceiling on papers per lookup, from the user's knob.
        max_costly_sites: Longest chain a sweep may diagonalise, from the user's
            knob. Held here rather than read inside the sweep so that the tool the
            agent calls and the plot the Lab draws obey the same ceiling -- they
            are the same function, and a limit honoured in one place only is a
            setting that appears to work until you check the other page.
        sweep_points: Resolution the agent's sweeps run at, from the same knob and
            already clipped for cost by
            :meth:`src.agent.setting.PhysicsSetting.points_for`, so the curve the
            agent draws and the curve the Lab draws come out at the same
            resolution. Both the default and the ceiling, exactly as
            :attr:`max_papers` is for a paper search: the model may ask for a
            coarser curve, and asking for a finer one is clipped rather than
            obeyed, because how much diagonalisation this machine does is the
            user's decision and not the model's.

            Deciding it here rather than in the tool schema is what makes the
            setting reach the agent at all. A default written into the schema is
            just a number in a paragraph the model reads -- and asked to fill the
            field in anyway, it copies whatever number that paragraph names. The
            first version of this said "at most 121" and got 121 back on the very
            first live question, on a knob set to 31.
        fetch: How to reach arXiv. Injected by the tests.
        wiki_fetch: How to reach Wikipedia. Injected by the tests.
        web_search: How to reach the search endpoint. Injected by the tests.
        mcp_transport: How to reach the MCP server. Injected by the tests.
    """

    spec: TFIMSpec
    use_comparison: bool = True
    use_sweep: bool = True
    use_arxiv: bool = True
    use_wikipedia: bool = True
    use_web: bool = False
    use_mcp: bool = False
    max_papers: int = 3
    max_costly_sites: int = MAX_COSTLY_SITES
    sweep_points: int = DEFAULT_POINTS
    fetch: Fetch | None = None
    wiki_fetch: Fetcher | None = None
    web_search: Search | None = None
    mcp_transport: Transport | None = None

    @property
    def web_available(self) -> bool:
        """Whether web search is both wanted and possible.

        Returns:
            ``True`` when the user asked for it and there is somewhere to ask. A
            test's injected search counts as configuration, which is what lets the
            tool be exercised without an endpoint.
        """
        return self.use_web and (self.web_search is not None or websearch.is_configured())

    @property
    def mcp_available(self) -> bool:
        """Whether a remote MCP server is both wanted and reachable.

        Returns:
            ``True`` when the user asked for it and a server is configured or a
            transport was injected.
        """
        return self.use_mcp and (self.mcp_transport is not None or mcp.is_configured())

    @property
    def external_available(self) -> bool:
        """Whether anything outside this project's own material is offered.

        The solver tools are always available and are not what this asks about: they
        compute from the same chain the answer is already about. What matters to
        :func:`src.agent.deciding.is_forced` is whether there is a *source* beyond
        the corpus to reach for, because that is what turns "compose what I have"
        from the only option into a choice.

        Returns:
            ``True`` when at least one outside source is enabled and reachable.
        """
        return bool(
            self.use_arxiv or self.use_wikipedia or self.web_available or self.mcp_available
        )

    @property
    def schemas(self) -> tuple[type[BaseModel], ...]:
        """The tool schemas to advertise to the model, in a fixed order.

        Returns:
            The enabled tools as Pydantic classes, verified physics first and the
            least trustworthy source last -- the order the model reads them in.
            Binding these is what turns a docstring into a tool description and a
            field into a JSON-schema property, so the text the model reads and the
            validation the arguments face are generated from one declaration and
            cannot drift apart.
        """
        offered: list[type[BaseModel]] = []
        if self.use_comparison:
            offered.append(CompareChain)
        if self.use_sweep:
            offered.append(SweepField)
        if self.use_arxiv:
            offered.append(FindPapers)
        if self.use_wikipedia:
            offered.append(LookUpWikipedia)
        if self.web_available:
            offered.append(SearchWeb)
        if self.mcp_available:
            offered.append(CallMcpTool)
        return tuple(offered)

    @property
    def names(self) -> tuple[str, ...]:
        """Names of the enabled tools, as the model will call them."""
        return tuple(schema.__name__ for schema in self.schemas)

    def run(self, tool: str, arguments: dict[str, Any]) -> ToolRun:
        """Validate one requested call and execute it.

        Args:
            tool: The tool name the model asked for.
            arguments: The arguments it supplied.

        Returns:
            The record. An unknown name, arguments that fail validation and a tool
            that declined the work are three different sentences and one control
            flow, because all three end the same way: the answer proceeds without
            this call.
        """
        lookup = {schema.__name__: schema for schema in self.schemas}
        schema = lookup.get(tool)
        if schema is None:
            offered = ", ".join(self.names) or "none"
            return ToolRun(
                tool=tool,
                arguments=arguments,
                ok=False,
                detail=f"there is no tool called {tool!r}; available: {offered}",
            )
        try:
            request = schema.model_validate(arguments)
        except Exception as error:
            return ToolRun(
                tool=tool,
                arguments=arguments,
                ok=False,
                detail=f"the arguments were not valid for {tool} ({type(error).__name__})",
            )
        if isinstance(request, CompareChain):
            run = compare_chain(self.spec, request.n_sites)
            return ToolRun(
                tool=tool,
                arguments=arguments,
                ok=run.ok,
                detail=run.detail,
                chain=run,
            )
        if isinstance(request, SweepField):
            # Omitted means no preference, which is the ordinary case and the one
            # the user's setting should decide. A number is taken as a request for
            # something coarser and clipped if it is not -- the same rule
            # `max_papers` applies a few lines down, and for the same reason: the
            # ceiling on what this machine spends belongs to whoever set it.
            asked = request.points
            wanted = self.sweep_points if asked is None else min(asked, self.sweep_points)
            curve = sweep_field(self.spec, wanted, request.observable, self.max_costly_sites)
            return ToolRun(
                tool=tool,
                arguments=arguments,
                ok=curve.ok,
                detail=curve.detail,
                sweep=curve,
            )
        if isinstance(request, FindPapers):
            search = find_papers(
                request.query,
                min(request.limit, self.max_papers),
                author=request.author,
                title=request.title,
                fetch=self.fetch,
            )
            return ToolRun(
                tool=tool,
                arguments=arguments,
                ok=search.found,
                detail=search.detail,
                papers=search,
            )
        if isinstance(request, LookUpWikipedia):
            found = look_up(
                request.query,
                depth="deep" if request.deep else "summary",
                fetch=self.wiki_fetch,
            )
            return ToolRun(
                tool=tool,
                arguments=arguments,
                ok=found.found,
                detail=found.detail or found.explain(),
                wiki=found,
            )
        if isinstance(request, SearchWeb):
            hits = search_web(request.query, request.limit, search=self.web_search)
            return ToolRun(
                tool=tool,
                arguments=arguments,
                ok=hits.found,
                detail=hits.detail,
                web=hits,
            )
        if isinstance(request, CallMcpTool):
            server = McpServer(url=mcp.endpoint() or "", transport=self.mcp_transport)
            call = server.call(request.tool, request.query)
            return ToolRun(
                tool=tool,
                arguments=arguments,
                ok=call.ok,
                detail=call.detail or call.explain(),
                remote=call,
            )
        # Unreachable while every schema in `schemas` has a branch above. Kept
        # because that is exactly the invariant a fourth tool would break, and
        # breaking it here is a recorded refusal rather than a fall through the
        # end of a function returning None.
        return ToolRun(
            tool=tool,
            arguments=arguments,
            ok=False,
            detail=f"{tool} is advertised but not wired up; this is a bug",
        )


def requested_calls(message: object) -> list[dict[str, Any]]:
    """Read the tool calls off a model reply, defensively.

    Args:
        message: Whatever the bound model returned.

    Returns:
        The calls, each as a mapping with at least ``name`` and ``args``. Empty
        when the model asked for nothing, which is the common and correct case --
        and equally when the reply has no tool-call field at all, since a provider
        that ignores the binding must not look like a crash.
    """
    calls = getattr(message, "tool_calls", None)
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, dict) and call.get("name")]


def consult(
    material: str,
    toolbox: Toolbox,
    *,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
) -> tuple[ToolRun, ...]:
    """Offer the tools, and run whatever the model asks for.

    Args:
        material: What has been established so far -- the question, the chain, the
            verified numbers and any retrieved passages. Wrapped as data, since
            the retrieved part of it was written by somebody else.
        toolbox: The tools to offer and the run they belong to.
        model: An explicit chat model, normally supplied only by tests.
        settings: Configuration to build a model from.

    Returns:
        One record per executed call, in the order the model asked. Empty when
        there are no tools, when no model is reachable, when the call failed, or
        when the model asked for nothing -- four situations with one consequence,
        which is that the answer is composed from what it already had.
    """
    if not toolbox.schemas:
        return ()
    resolved = chat_model_or_none(model, settings)
    if resolved is None:
        return ()
    messages = [SystemMessage(content=CONSULT_SYSTEM), HumanMessage(content=as_data(material))]
    name = getattr(resolved, "model_name", None) or type(resolved).__name__
    try:
        bound = resolved.bind_tools(list(toolbox.schemas))
        with log_llm_call(name, purpose="consult") as call:
            reply = bound.invoke(messages)
            call.observe(reply)
    except Exception as error:
        LOG.warning("consult_failed", extra={"error_type": type(error).__name__})
        return ()

    calls = requested_calls(reply)
    runs = [toolbox.run(str(call["name"]), dict(call.get("args") or {})) for call in calls][
        :MAX_TOOL_CALLS
    ]
    if len(calls) > MAX_TOOL_CALLS:
        runs.append(
            ToolRun(
                tool="(dropped)",
                arguments={},
                ok=False,
                detail=(
                    f"the model asked for {len(calls)} tool calls; "
                    f"{MAX_TOOL_CALLS} were run and the rest were dropped"
                ),
            )
        )
    LOG.info(
        "consulted",
        extra={"asked": len(calls), "ran": len(runs), "tools": list(toolbox.names)},
    )
    return tuple(runs)


def context_for(runs: tuple[ToolRun, ...]) -> str:
    """Render what the tools produced, for the narrator's prompt.

    Args:
        runs: The executed calls.

    Returns:
        A block per useful result: comparison chains as verified numbers, papers
        as explicitly unverified further reading. Empty when nothing usable came
        back, so a failed lookup adds no text at all rather than an apology the
        narrator would then have to explain.
    """
    blocks: list[str] = []
    checks = [
        run.chain.check for run in runs if run.chain is not None and run.chain.check is not None
    ]
    if checks:
        blocks.append(
            "\n".join(
                ["ADDITIONAL CHAINS, COMPUTED AND CROSS-CHECKED THE SAME WAY:"]
                + [check.summary() for check in checks]
            )
        )
    for run in runs:
        if run.sweep is not None and run.sweep.ok:
            blocks.append(run.sweep.table())
        if run.papers is not None and run.papers.found:
            blocks.append(run.papers.context())
        if run.wiki is not None and run.wiki.found:
            blocks.append(run.wiki.context())
        if run.web is not None and run.web.found:
            blocks.append(run.web.context())
        if run.remote is not None and run.remote.ok:
            blocks.append(run.remote.context())
    return "\n\n".join(blocks)
