"""One place that reads the environment.

Written because the same bug shipped twice: `demo_agent.py --live` failed with
"No ANTHROPIC_API_KEY" while the key sat in `.env`, and then the API silently
fell back to an empty FakeLLM and told the user its own test-harness error.

The cause both times was that `load_dotenv()` lived in whichever entry point
happened to remember it. Config loading scattered across entry points is config
loading that is wrong in at least one of them.

`settings()` is cached, so importing it from anywhere is free and every caller
sees the same values.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    atlassian_base_url: str = ""
    atlassian_email: str = ""
    atlassian_api_token: str = ""
    jira_target_release_field: str = ""
    jira_project_keys: tuple[str, ...] = ()
    jira_allow_writes: bool = False
    risk_clock_offset_days: int = 0

    @property
    def has_llm(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_jira(self) -> bool:
        """Enough to run a real sync at startup.

        Project keys are part of it, not an optional extra. Without them there
        is nothing to search, and a sync over zero projects returns a
        `CompleteSync` with no findings — "nothing is at risk", stated with
        confidence, because nobody was asked. The missing config must fail
        loudly enough to stay on the sandbox instead.
        """
        return bool(
            self.atlassian_base_url
            and self.atlassian_email
            and self.atlassian_api_token
            and self.jira_project_keys
        )

    @property
    def can_write_to_jira(self) -> bool:
        """Two independent conditions, and both are required.

        Credentials alone must never imply permission to write. A tenant is
        configured for reading first — that is how a pilot starts — and the day
        someone adds project keys should not be the day the service gains the
        ability to edit tickets. `JIRA_ALLOW_WRITES` is the separate, deliberate
        act.
        """
        return self.has_jira and self.jira_allow_writes

    @property
    def has_embeddings(self) -> bool:
        return bool(self.voyage_api_key)

    def missing_for_live(self) -> list[str]:
        missing = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        return missing


def _keys(raw: str) -> tuple[str, ...]:
    """Parse JIRA_PROJECT_KEYS=INS, CLM,PLAT.

    Trailing commas and stray spaces are normal in a hand-edited .env, and a
    project key of "" produces a JQL that 400s with a message about the query
    rather than about the config.
    """
    return tuple(key.strip().upper() for key in raw.split(",") if key.strip())


# Only these count as permission. `bool(os.getenv(...))` would read "false" and
# "0" as True, which is how a flag guarding writes to a client's tenant gets
# turned on by someone trying to turn it off.
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _flag(raw: str) -> bool:
    return raw.strip().lower() in _TRUTHY


def _offset(raw: str) -> int:
    """Days to shift the clock the risk rules are evaluated against.

    Stale detection cannot be seeded — Jira will not backdate `created`,
    comments or changelog entries — so on a freshly seeded sandbox the rule the
    client cares most about finds nothing. The alternative people reach for is
    lowering `stale_days_high` until something fires, which means shipping
    thresholds tuned to fake data. The rules take `now` as a parameter so that
    the clock can move instead and the thresholds stay honest.

    Forward only. A negative offset would evaluate in the past and *hide* real
    findings, which has no legitimate use and one obvious illegitimate one.
    Unparseable values fall back to 0 rather than raising: a typo in a demo
    setting must not stop the service from booting.
    """
    try:
        return max(0, int(raw.strip()))
    except (TypeError, ValueError):
        return 0


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Read configuration once.

    `load_dotenv` is imported inside the function rather than at module scope
    so a test can neuter it. That matters more than it looks: `delenv` removes
    a variable and `load_dotenv()` reads it straight back from the developer's
    .env, so a test asserting "no key configured" passes on a machine without a
    .env and fails on one with it — testing the machine, not the app.
    """
    try:
        import dotenv

        dotenv.load_dotenv()
    except ImportError:
        # python-dotenv is optional; real deployments inject env directly.
        pass

    loaded = Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        atlassian_base_url=os.getenv("ATLASSIAN_BASE_URL", ""),
        atlassian_email=os.getenv("ATLASSIAN_EMAIL", ""),
        atlassian_api_token=os.getenv("ATLASSIAN_API_TOKEN", ""),
        jira_target_release_field=os.getenv("JIRA_TARGET_RELEASE_FIELD", ""),
        jira_project_keys=_keys(os.getenv("JIRA_PROJECT_KEYS", "")),
        jira_allow_writes=_flag(os.getenv("JIRA_ALLOW_WRITES", "")),
        risk_clock_offset_days=_offset(os.getenv("RISK_CLOCK_OFFSET_DAYS", "")),
    )

    if not loaded.has_llm:
        # Loud, and it says what the consequence is. The previous behaviour was
        # a silent fallback to FakeLLM, which surfaced to the user as a test
        # harness error about "scripted responses" — an internal detail that
        # told them nothing about the missing key.
        logger.warning(
            "ANTHROPIC_API_KEY is not set. The service will run but cannot "
            "answer questions."
        )
    return loaded
