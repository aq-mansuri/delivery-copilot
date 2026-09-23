"""Tests for tracing, cost attribution and latency reporting."""

from __future__ import annotations

import asyncio
import json
import pytest

from app.agent.llm import FakeLLM, LLMResponse, text_response
from app.core.tracing import Rates, TracedLLM, Trace, compare


class TestSpans:
    def test_spans_nest_via_context(self):
        trace = Trace()
        with trace.span("outer", "node"):
            with trace.span("inner", "llm"):
                pass
        outer, inner = trace.spans
        assert outer.parent_id is None
        assert inner.parent_id == outer.span_id

    def test_errors_are_recorded_then_reraised(self):
        """A tracer that swallows exceptions turns a visible failure into a
        mysterious one."""
        trace = Trace()
        with pytest.raises(ValueError):
            with trace.span("failing", "tool"):
                raise ValueError("boom")
        assert "ValueError: boom" in trace.spans[0].error
        assert trace.errors()

    def test_span_closes_even_on_error(self):
        trace = Trace()
        with pytest.raises(RuntimeError):
            with trace.span("x", "tool"):
                raise RuntimeError("x")
        assert trace.spans[0].ended_at is not None

    async def test_concurrent_traces_do_not_interleave(self):
        """A module-level global breaks the moment two questions run at once,
        which is what a FastAPI deployment does."""

        async def run(trace, name):
            with trace.span(name, "node"):
                await asyncio.sleep(0.01)
                with trace.span(f"{name}.child", "llm"):
                    await asyncio.sleep(0.01)

        a, b = Trace(), Trace()
        await asyncio.gather(run(a, "a"), run(b, "b"))

        for trace in (a, b):
            root = [s for s in trace.spans if s.parent_id is None]
            assert len(root) == 1
            child = [s for s in trace.spans if s.parent_id is not None]
            assert child[0].parent_id == root[0].span_id


class TestCost:
    def test_cost_uses_injected_rates(self):
        """A stale hardcoded rate produces confidently wrong cost reporting."""
        trace = Trace(rates=Rates(input_per_mtok=3.0, output_per_mtok=15.0))
        with trace.span("call", "llm") as span:
            span.input_tokens = 1_000_000
            span.output_tokens = 100_000
        assert trace.total_cost == pytest.approx(3.0 + 1.5)

    def test_cost_is_attributed_per_kind(self):
        """Not a monthly bill — which node spent it."""
        trace = Trace()
        with trace.span("answer", "llm") as s:
            s.input_tokens, s.output_tokens = 5000, 500
        for _ in range(12):
            with trace.span("judge", "judge") as s:
                s.input_tokens, s.output_tokens = 800, 80

        by_kind = trace.by_kind()
        assert by_kind["judge"]["count"] == 12
        assert by_kind["judge"]["cost"] > by_kind["llm"]["cost"]

    async def test_traced_llm_records_usage(self):
        trace = Trace()
        inner = FakeLLM(responses=[
            LLMResponse(text="hi", input_tokens=3000, output_tokens=200)
        ])
        traced = TracedLLM(inner, trace)
        await traced.complete(system="s", messages=[{"role": "user", "content": "q"}])

        assert trace.total_input_tokens == 3000
        assert trace.total_output_tokens == 200
        assert trace.total_cost > 0

    async def test_fake_llm_is_traced_identically(self):
        """Tracing has to be testable offline, so it wraps rather than edits."""
        trace = Trace()
        traced = TracedLLM(FakeLLM(responses=[text_response("x")]), trace)
        await traced.complete(system="s", messages=[])
        assert trace.spans[0].kind == "llm"


class TestLatency:
    def test_wall_time_is_the_root_not_the_sum(self):
        """Summing double-counts nesting and reports more time than passed."""
        trace = Trace()
        with trace.span("root", "node"):
            with trace.span("a", "llm"):
                pass
            with trace.span("b", "llm"):
                pass
        total_of_all = sum(s.duration_ms for s in trace.spans)
        assert trace.wall_ms < total_of_all

    def test_slowest_spans_are_identifiable(self):
        """'It takes nine seconds' is not actionable."""
        trace = Trace()
        with trace.span("fast", "retrieval"):
            pass
        with trace.span("slow", "llm"):
            import time

            time.sleep(0.02)
        assert trace.slowest(1)[0].name == "slow"


class TestExport:
    def test_dict_export_round_trips(self, tmp_path):
        trace = Trace(question="what is at risk?")
        with trace.span("answer", "llm") as s:
            s.input_tokens, s.output_tokens = 100, 10
        path = tmp_path / "trace.json"
        trace.save(path)

        loaded = json.loads(path.read_text())
        assert loaded["question"] == "what is at risk?"
        assert loaded["spans"][0]["input_tokens"] == 100

    def test_otel_shape_carries_usage(self):
        trace = Trace()
        with trace.span("answer", "llm") as s:
            s.input_tokens = 500
        span = trace.to_otel_spans()[0]
        assert span["attributes"]["gen_ai.usage.input_tokens"] == 500
        assert span["endTimeUnixNano"] >= span["startTimeUnixNano"]

    def test_otel_status_reflects_errors(self):
        trace = Trace()
        with pytest.raises(ValueError):
            with trace.span("x", "tool"):
                raise ValueError("bad")
        assert trace.to_otel_spans()[0]["status"]["code"] == 2


class TestComparison:
    def test_percentiles_expose_the_tail(self):
        """A p95 triple the mean usually means one path a client hits in the
        first hour."""
        traces = []
        for tokens in [1000] * 19 + [20000]:
            t = Trace()
            with t.span("call", "llm") as s:
                s.input_tokens = tokens
            traces.append(t)

        report = compare(traces)
        assert "p95" in report and "max" in report

    def test_empty_input_does_not_crash(self):
        assert compare([]) == "no traces"


class TestSelfTime:
    """Percentages exceeded 100% before this existed: a parent's duration
    includes its children, so summing by kind double-counted."""

    def test_kind_shares_do_not_exceed_wall_time(self):
        trace = Trace()
        with trace.span("outer", "node"):
            import time

            with trace.span("inner", "node"):
                time.sleep(0.01)
        total_share = sum(s["self_ms"] for s in trace.by_kind().values())
        assert total_share <= trace.wall_ms * 1.01

    def test_parent_self_time_excludes_children(self):
        trace = Trace()
        import time

        with trace.span("parent", "node"):
            with trace.span("child", "llm"):
                time.sleep(0.02)
        parent = trace.spans[0]
        assert trace.self_ms(parent) < parent.duration_ms / 2

    def test_leaf_self_time_equals_duration(self):
        trace = Trace()
        with trace.span("leaf", "llm"):
            pass
        leaf = trace.spans[0]
        assert trace.self_ms(leaf) == pytest.approx(leaf.duration_ms)

    def test_export_includes_self_time(self):
        trace = Trace()
        with trace.span("a", "node"):
            with trace.span("b", "llm"):
                pass
        assert "self_ms" in trace.to_dict()["spans"][0]


class TestOccupancy:
    """Self-time fixed nesting. Once judging ran concurrently, ten overlapping
    spans reported a 213% share — work divided by clock, called a share."""

    async def test_concurrent_spans_occupy_less_than_their_sum(self):
        import asyncio

        trace = Trace()

        async def work():
            with trace.span("call", "judge"):
                await asyncio.sleep(0.05)

        with trace.span("root", "node"):
            await asyncio.gather(*(work() for _ in range(5)))

        cumulative = sum(s.duration_ms for s in trace.spans if s.kind == "judge")
        occupancy = trace.occupancy_ms("judge")
        assert cumulative > occupancy * 2
        assert occupancy <= trace.wall_ms * 1.05

    async def test_share_never_exceeds_one_hundred_percent(self):
        import asyncio

        trace = Trace()

        async def work():
            with trace.span("call", "judge"):
                await asyncio.sleep(0.03)

        with trace.span("root", "node"):
            await asyncio.gather(*(work() for _ in range(8)))

        for stats in trace.by_kind().values():
            share = stats["occupancy_ms"] / trace.wall_ms * 100
            assert share <= 101

    async def test_concurrency_factor_reflects_parallelism(self):
        import asyncio

        trace = Trace()

        async def work():
            with trace.span("call", "judge"):
                await asyncio.sleep(0.04)

        with trace.span("root", "node"):
            await asyncio.gather(*(work() for _ in range(4)))

        assert trace.concurrency_factor("judge") > 2.5

    def test_sequential_spans_have_factor_one(self):
        trace = Trace()
        import time

        for _ in range(3):
            with trace.span("call", "llm"):
                time.sleep(0.005)
        assert trace.concurrency_factor("llm") == pytest.approx(1.0, abs=0.15)

    def test_missing_kind_returns_zero(self):
        assert Trace().occupancy_ms("nope") == 0.0
        assert Trace().concurrency_factor("nope") == 0.0


class TestPerModelRates:
    """A single rate pair applied to every span priced Haiku tokens as Sonnet
    tokens, so switching to a cheaper judge made cost_per_eval go UP."""

    def test_each_span_uses_its_own_model_rate(self):
        from app.core.tracing import rates_for

        trace = Trace()
        with trace.span("answer", "llm") as s:
            s.model, s.input_tokens = "claude-sonnet-4-6", 1_000_000
        with trace.span("judge", "judge") as s:
            s.model, s.input_tokens = "claude-haiku-4-5-20251001", 1_000_000

        assert trace.spans[0].cost() == pytest.approx(3.0)
        assert trace.spans[1].cost() == pytest.approx(0.8)
        assert rates_for("claude-haiku-x").input_per_mtok < rates_for(
            "claude-sonnet-x"
        ).input_per_mtok

    def test_unknown_model_falls_back_rather_than_zeroing(self):
        """A zero disappears from a budget check silently."""
        from app.core.tracing import rates_for

        assert rates_for("some-other-model").input_per_mtok > 0

    async def test_traced_llm_records_the_model(self):
        class Fake:
            model = "claude-haiku-4-5-20251001"

            async def complete(self, **kwargs):
                from app.agent.llm import LLMResponse

                return LLMResponse(text="x", input_tokens=1000)

        trace = Trace()
        await TracedLLM(Fake(), trace, kind="judge").complete(system="s", messages=[])
        assert trace.spans[0].model == "claude-haiku-4-5-20251001"

    def test_mixed_model_trace_sums_correctly(self):
        trace = Trace()
        with trace.span("a", "llm") as s:
            s.model, s.input_tokens = "claude-sonnet-4-6", 1_000_000
        with trace.span("b", "judge") as s:
            s.model, s.input_tokens = "claude-haiku-4-5-20251001", 1_000_000
        assert trace.total_cost == pytest.approx(3.8)
