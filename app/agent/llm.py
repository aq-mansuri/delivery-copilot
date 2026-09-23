"""LLM access, behind a protocol.

Same reasoning as the embedder: the agent depends on an interface, not on
Anthropic's SDK. That keeps every test in this project runnable with no API key,
no network and no bill — which matters more here than for embeddings, because
agent tests exercise multi-step flows that would otherwise cost real money on
every CI run.

The protocol is deliberately thin: one call, returning text and any tool uses.
Everything above it — retries, the graph, approval — is our code, not the SDK's.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"

# Grading with the same model that answered correlates errors: a model that
# misreads a passage while answering tends to misread it the same way while
# judging. A different, cheaper model decorrelates that AND cuts the eval bill —
# measured, judging was ~5x the cost of the answer it graded.
#
# Verify against calibration before switching. A cheaper judge that misses
# overreach produces a groundedness score measuring its own leniency, and
# `calibrate()` exists precisely to catch that.
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_uses: tuple[ToolUse, ...] = ()
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0

    def wants_tools(self) -> bool:
        return bool(self.tool_uses)

    def cost_usd(
        self, input_per_mtok: float = 3.0, output_per_mtok: float = 15.0
    ) -> float:
        """Rough per-call cost, for the tracing work on Day 5.

        Rates are passed in rather than hardcoded because they change, and a
        stale constant buried in a class produces confidently wrong cost
        reporting — worse than none.
        """
        return (
            self.input_tokens * input_per_mtok
            + self.output_tokens * output_per_mtok
        ) / 1_000_000


class LLM(Protocol):
    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


class AnthropicLLM:
    """Anthropic Messages API.

    ## On temperature, and on writing against an SDK that moves

    This system answers delivery-risk questions at a regulated insurer, so the
    same question on the same data should give the same answer twice — sampling
    variance is a liability here, and it makes eval scores noisy enough to mask
    real regressions.

    `temperature` was a parameter when this was written and is not one in the
    installed SDK. Rather than hardcode either assumption, the accepted
    parameters are read off the SDK's own signature at construction and
    unsupported ones are dropped, with a warning.

    This is the honest way to depend on a moving API: adapt to what is actually
    installed, say so out loud, and do not pretend the guarantee still holds
    when it does not. Determinism now rests on the model's default behaviour
    rather than on an explicit setting — worth stating in the eval writeup
    rather than quietly assuming.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
    ):
        import inspect

        from anthropic import AsyncAnthropic
        from anthropic.resources.messages import AsyncMessages

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No ANTHROPIC_API_KEY. Use FakeLLM for offline work."
            )
        self.model = model
        self._client = AsyncAnthropic(api_key=key, max_retries=max_retries)

        try:
            self._accepted = set(
                inspect.signature(AsyncMessages.create).parameters
            )
        except (TypeError, ValueError):
            # Signature unreadable (C extension, heavy decoration). Send
            # everything and let the API reject what it does not want.
            self._accepted = set()

        if self._accepted and "temperature" not in self._accepted:
            logger.info(
                "Installed anthropic SDK does not accept `temperature`; "
                "relying on the model default. Determinism is not guaranteed "
                "by an explicit setting."
            )

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = list(tools)

        # Drop anything this SDK version does not accept. Without this, an SDK
        # upgrade turns a working agent into a TypeError mid-run.
        if self._accepted:
            kwargs = {k: v for k, v in kwargs.items() if k in self._accepted}

        response = await self._client.messages.create(**kwargs)

        # Content is a list of typed blocks, not a string. Indexing [0].text
        # works right up until the model leads with a tool_use block, which is
        # exactly when the agent path matters.
        text_parts = [b.text for b in response.content if b.type == "text"]
        tool_uses = tuple(
            ToolUse(id=b.id, name=b.name, input=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        )

        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_uses=tool_uses,
            stop_reason=response.stop_reason or "end_turn",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


@dataclass
class FakeLLM:
    """Scripted responses, in order, for deterministic tests.

    Records every call so tests can assert on what the agent actually sent —
    which system prompt, which tools were offered, whether prior tool results
    were carried forward. Those are the things that break silently.
    """

    responses: list[LLMResponse] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": [t["name"] for t in (tools or [])],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if not self.responses:
            raise AssertionError(
                f"FakeLLM ran out of scripted responses on call "
                f"{len(self.calls)}. The agent made more LLM calls than the "
                "test expected — usually a loop that is not terminating."
            )
        return self.responses.pop(0)

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1]["system"] if self.calls else ""


def text_response(text: str, **kwargs: Any) -> LLMResponse:
    return LLMResponse(text=text, **kwargs)


def tool_response(name: str, tool_input: dict[str, Any], *, text: str = "") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_uses=(ToolUse(id=f"tu_{name}", name=name, input=tool_input),),
        stop_reason="tool_use",
    )


def default_llm() -> LLM:
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicLLM()
    logger.warning("ANTHROPIC_API_KEY not set — using an empty FakeLLM.")
    return FakeLLM()
