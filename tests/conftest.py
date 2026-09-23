"""Shared test setup.

## Offline is enforced here, not promised in a README

"Every test runs offline" has been a rule since Day 1, and `FakeLLM` and
`FakeEmbedder` were built to keep it. It still broke: a `/sync` test called code
that branches on `settings().has_jira`, which is True on any machine with a
real `.env` — so the test read a live Jira tenant, passed on the developer's
laptop, and would have failed in CI with a credentials error that looks like an
infrastructure problem rather than a test bug.

The rule is the kind this project encodes in types rather than discipline, so it
is encoded here instead of restated: outbound sockets raise. A test that reaches
for the network fails immediately, on the machine that has credentials, naming
what it tried to do.

This costs nothing legitimate. `respx` intercepts httpx above the socket layer,
`TestClient` speaks ASGI in-process, and the fakes never had a socket to open.
"""

from __future__ import annotations

import socket

import pytest


class NetworkAccessInTest(RuntimeError):
    pass


def _refuse(*args, **kwargs):
    raise NetworkAccessInTest(
        "A test tried to open a network connection. Every test in this project "
        "runs offline — use FakeLLM, FakeEmbedder or respx. If the code under "
        "test branches on configuration (settings().has_jira, has_llm), pin the "
        "environment with a fixture; otherwise the test passes on a machine "
        "with a .env and fails everywhere else."
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
