"""Tracing: where the time and the money go.

Built in-process rather than starting with OpenTelemetry or Langfuse, for the
same reason the heuristic decomposer came before the LLM one. A client asks
"what does a question cost" on the first call, and the answer should not require
them to stand up a collector. `to_otel_spans()` exists for when it does.

Three things this has to get right, because they are the things asked about:

**Cost per question, attributable.** Not a monthly bill — which node spent it.
Claim-level judging makes one call per claim, so evaluating a twelve-claim
answer costs thirteen calls. That is invisible until it is measured, and it is
the kind of number that decides whether an eval runs per-commit or nightly.

**Latency by phase.** "It takes nine seconds" is not actionable. "Retrieval is
40ms, the model is 8.6s" is.

**Token counts on the input side.** Live runs showed 3,000-6,500 input tokens per
question. Context is the cost driver in a RAG system, and passage count is the
lever.

Rates are injected, never hardcoded. A stale constant buried in a class produces
confidently wrong cost reporting, which is worse than none.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rates:
    """USD per million tokens. Change with the price list, not with a code edit."""

    input_per_mtok: float = 3.0
    output_per_mtok: float = 15.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok
            + output_tokens * self.output_per_mtok
        ) / 1_000_000


# Rates differ per model, and a single pair applied to every span prices a cheap
# model's tokens as an expensive one's. Observed: switching the judge to Haiku
# made cost_per_eval go UP, because Haiku tokens were being charged at Sonnet
# rates — the number moved in the direction that hides the saving.
#
# Prefixes rather than exact ids, so a dated model string still matches. Verify
# against the current price list; these are not authoritative.
MODEL_RATES: dict[str, Rates] = {
    "claude-opus": Rates(15.0, 75.0),
    "claude-sonnet": Rates(3.0, 15.0),
    "claude-haiku": Rates(0.8, 4.0),
}

DEFAULT_RATES = Rates()


def rates_for(model: str | None) -> Rates:
    """Rates for a model id, falling back to the default.

    An unknown model gets the default rather than zero: reporting no cost for a
    model nobody mapped is worse than reporting an approximate one, because a
    zero disappears from a budget check silently.
    """
    if not model:
        return DEFAULT_RATES
    for prefix, rates in MODEL_RATES.items():
        if model.startswith(prefix):
            return rates
    logger.warning("no rates for model %r — using defaults", model)
    return DEFAULT_RATES


@dataclass
class Span:
    name: str
    kind: str  # "llm" | "tool" | "retrieval" | "node" | "judge"
    span_id: str
    parent_id: str | None
    started_at: float
    ended_at: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    model: str | None = None

    @property
    def duration_ms(self) -> float:
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at) * 1000

    def cost(self, rates: Rates | None = None) -> float:
        """Cost at this span's own model rates.

        The trace-level rates are only a fallback; a trace mixing an answering
        model and a cheaper judge has two prices in it.
        """
        effective = rates_for(self.model) if self.model else (rates or DEFAULT_RATES)
        return effective.cost(self.input_tokens, self.output_tokens)


_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_span", default=None
)


@dataclass
class Trace:
    """One question's worth of spans.

    A contextvar carries the parent id, so nesting works across `await` without
    threading a tracer argument through every function. The alternative — a
    module-level global — breaks the moment two questions are handled
    concurrently, which is exactly what a FastAPI deployment does.
    """

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    question: str = ""
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    spans: list[Span] = field(default_factory=list)
    rates: Rates = field(default_factory=Rates)

    @contextmanager
    def span(self, name: str, kind: str, **attributes: Any) -> Iterator[Span]:
        span = Span(
            name=name,
            kind=kind,
            span_id=uuid.uuid4().hex[:12],
            parent_id=_current_span.get(),
            started_at=time.perf_counter(),
            attributes=dict(attributes),
        )
        self.spans.append(span)
        token = _current_span.set(span.span_id)
        try:
            yield span
        except Exception as exc:
            # Recorded, then re-raised. A tracer that swallows exceptions turns
            # a visible failure into a mysterious one.
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.ended_at = time.perf_counter()
            _current_span.reset(token)

    # -- aggregates ---------------------------------------------------------

    @property
    def total_cost(self) -> float:
        return sum(s.cost(self.rates) for s in self.spans)

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.spans)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.spans)

    @property
    def wall_ms(self) -> float:
        """Elapsed time for the root span, not the sum of all spans.

        Summing double-counts every nested span and, once anything runs
        concurrently, reports more time than actually passed.
        """
        roots = [s for s in self.spans if s.parent_id is None]
        return max((s.duration_ms for s in roots), default=0.0)

    def self_ms(self, span: Span) -> float:
        """Time spent in this span excluding its children.

        The standard profiler notion, and necessary here: a parent's duration
        includes everything nested inside it, so summing raw durations by kind
        double-counts and reports shares above 100%. Observed as
        "node 2x 4ms (118%)" before this existed.

        Self-time is also the actionable number — a node whose total is 9s but
        whose self-time is 3ms is not where the latency lives.
        """
        children = sum(
            child.duration_ms
            for child in self.spans
            if child.parent_id == span.span_id
        )
        return max(span.duration_ms - children, 0.0)

    def occupancy_ms(self, kind: str) -> float:
        """Wall-clock time during which at least one span of this kind was open.

        Self-time fixed double-counting from NESTING. It does not fix
        CONCURRENCY: ten judge calls running at once have ten real durations
        that sum to more than the elapsed time, which produced a reported share
        of 213%.

        Two different questions need two different numbers:

          cumulative  how much work was done      (may exceed wall)
          occupancy   how much clock it owned     (never exceeds wall)

        Occupancy merges overlapping intervals, so a share computed from it is
        a share of real time.
        """
        intervals = sorted(
            (s.started_at, s.ended_at or s.started_at)
            for s in self.spans
            if s.kind == kind
        )
        if not intervals:
            return 0.0

        total = 0.0
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                total += current_end - current_start
                current_start, current_end = start, end
        total += current_end - current_start
        return total * 1000

    def concurrency_factor(self, kind: str) -> float:
        """Average spans of this kind in flight while any were running.

        1.0 means fully sequential. A factor well below the configured
        concurrency limit means raising the limit will not help — something
        else is serialising the work.
        """
        occupancy = self.occupancy_ms(kind)
        if not occupancy:
            return 0.0
        cumulative = sum(
            s.duration_ms for s in self.spans if s.kind == kind
        )
        return cumulative / occupancy

    def by_kind(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for span in self.spans:
            bucket = out.setdefault(
                span.kind,
                {"count": 0, "ms": 0.0, "self_ms": 0.0, "cost": 0.0, "tokens": 0},
            )
            bucket["count"] += 1
            bucket["ms"] += span.duration_ms
            bucket["self_ms"] += self.self_ms(span)
            bucket["cost"] += span.cost(self.rates)
            bucket["tokens"] += span.input_tokens + span.output_tokens

        for kind, bucket in out.items():
            bucket["occupancy_ms"] = self.occupancy_ms(kind)
            bucket["concurrency"] = self.concurrency_factor(kind)
        return out

    def llm_calls(self) -> list[Span]:
        return [s for s in self.spans if s.kind in ("llm", "judge")]

    def errors(self) -> list[Span]:
        return [s for s in self.spans if s.error]

    def summary(self) -> str:
        lines = [
            f"trace {self.trace_id}   {self.wall_ms:.0f}ms   "
            f"${self.total_cost:.4f}   "
            f"{self.total_input_tokens} in / {self.total_output_tokens} out   "
            f"{len(self.llm_calls())} model call(s)"
        ]
        lines.append(
            f"  {'kind':<10} {'n':>3}   {'work':>8}  {'clock':>8} "
            f"{'share':>6} {'conc':>5}   cost"
        )
        for kind, stats in sorted(
            self.by_kind().items(), key=lambda kv: -kv[1]["occupancy_ms"]
        ):
            # Share comes from occupancy — clock time this kind owned — so it
            # cannot exceed 100% however much work ran in parallel.
            share = (
                stats["occupancy_ms"] / self.wall_ms * 100 if self.wall_ms else 0
            )
            lines.append(
                f"  {kind:<10} {int(stats['count']):>3}x  "
                f"{stats['ms']:>7.0f}ms  {stats['occupancy_ms']:>7.0f}ms "
                f"{share:>5.0f}% {stats['concurrency']:>4.1f}x   "
                f"${stats['cost']:.4f}"
            )
        for span in self.errors():
            lines.append(f"  ERROR in {span.name}: {span.error}")
        return "\n".join(lines)

    def slowest(self, limit: int = 5) -> list[Span]:
        return sorted(self.spans, key=lambda s: -s.duration_ms)[:limit]

    # -- export -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "started_at": self.started_at.isoformat(),
            "wall_ms": round(self.wall_ms, 1),
            "cost_usd": round(self.total_cost, 6),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "spans": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "span_id": s.span_id,
                    "parent_id": s.parent_id,
                    "duration_ms": round(s.duration_ms, 2),
                    "self_ms": round(self.self_ms(s), 2),
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "cost_usd": round(s.cost(self.rates), 6),
                    "model": s.model,
                    "attributes": s.attributes,
                    "error": s.error,
                }
                for s in self.spans
            ],
        }

    def to_otel_spans(self) -> list[dict[str, Any]]:
        """Shape an OTLP exporter expects, for when a client wants their own
        collector. Semantic conventions for LLM attributes are still settling,
        so `gen_ai.*` names may need revisiting."""
        return [
            {
                "name": s.name,
                "spanId": s.span_id,
                "parentSpanId": s.parent_id,
                "startTimeUnixNano": int(s.started_at * 1e9),
                "endTimeUnixNano": int((s.ended_at or s.started_at) * 1e9),
                "attributes": {
                    "gen_ai.operation.name": s.kind,
                    "gen_ai.usage.input_tokens": s.input_tokens,
                    "gen_ai.usage.output_tokens": s.output_tokens,
                    **s.attributes,
                },
                "status": {"code": 2 if s.error else 1, "message": s.error},
            }
            for s in self.spans
        ]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


class TracedLLM:
    """Wraps any LLM and records a span per call.

    A wrapper rather than edits inside `AnthropicLLM`, so `FakeLLM` is traced
    identically and tracing stays testable offline.
    """

    def __init__(self, inner, trace: Trace, *, kind: str = "llm"):
        self._inner = inner
        self._trace = trace
        self._kind = kind
        # Read off the wrapped client so cost uses the right price list without
        # the caller having to remember to pass it.
        self._model = getattr(inner, "model", None)

    async def complete(self, **kwargs: Any):
        tools = kwargs.get("tools") or []
        with self._trace.span(
            "llm.complete",
            self._kind,
            tool_count=len(tools),
            max_tokens=kwargs.get("max_tokens"),
        ) as span:
            span.model = self._model
            response = await self._inner.complete(**kwargs)
            span.input_tokens = response.input_tokens
            span.output_tokens = response.output_tokens
            span.attributes["stop_reason"] = response.stop_reason
            span.attributes["tool_uses"] = len(response.tool_uses)
            return response


def compare(traces: list[Trace]) -> str:
    """Cost and latency across several runs.

    Averages alone hide the tail. A p95 that is triple the mean usually means
    one path — a retry, an extra tool round — that a client will hit in the
    first hour.
    """
    if not traces:
        return "no traces"

    costs = sorted(t.total_cost for t in traces)
    times = sorted(t.wall_ms for t in traces)

    def pct(values: list[float], p: float) -> float:
        return values[min(int(len(values) * p), len(values) - 1)]

    return "\n".join(
        [
            f"{len(traces)} traces",
            f"  cost    mean ${sum(costs) / len(costs):.4f}   "
            f"p50 ${pct(costs, 0.5):.4f}   p95 ${pct(costs, 0.95):.4f}   "
            f"max ${costs[-1]:.4f}",
            f"  latency mean {sum(times) / len(times):.0f}ms   "
            f"p50 {pct(times, 0.5):.0f}ms   p95 {pct(times, 0.95):.0f}ms   "
            f"max {times[-1]:.0f}ms",
        ]
    )
