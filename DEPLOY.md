# Deploying

## Run it locally

    cp .env.example .env        # fill in ANTHROPIC_API_KEY
    docker compose up --build

Then http://localhost:8000. `/health` reports `can_answer: false` and names the
missing variable if the key is absent — the service starts either way, because
approvals still work without a model.

## What is production-ready and what is not

Being straight about this matters more than a green checkmark. A deployment
guide that implies more than the code delivers is how a client discovers a
limitation during a demo.

| | State |
|---|---|
| Stateless request handling | Ready |
| Non-root container, no build toolchain in the image | Ready |
| Secrets from environment, never baked in | Ready |
| Health check that verifies the index, not just the port | Ready |
| **Approvals survive a restart** | **No — `ApprovalStore` is in memory** |
| **Vector index persists** | **No — rebuilt per process** |
| Authentication | No — `X-Actor` is self-asserted |
| Rate limiting | No |
| Corpus refresh from a live tenant | No — the seed corpus is compiled in |

The first two are the blockers for anything beyond a pilot. `ApprovalStore`
already has the right shape — proposals in, decisions out, audit durable before
apply — so the change is a Postgres-backed implementation of the same
interface, not a redesign. The commented-out `db` service in `compose.yaml` is
the placeholder.

Authentication matters more than it looks: `X-Actor` is whatever the caller
sends, so the audit trail currently records a claim rather than an identity. In
a regulated insurer that is the difference between an audit trail and a log.
Behind an SSO proxy that sets the header from a verified session, it becomes
real.

## Deploying to a host

The image is a standard uvicorn service on port 8000 with no local state, so
anything that runs a container works: Fly, Render, Railway, Cloud Run, ECS.

    docker build -t delivery-copilot .
    docker run -p 8000:8000 --env-file .env delivery-copilot

Two settings that are easy to miss:

**Proxy buffering.** SSE dies behind a proxy that buffers. The app sets
`X-Accel-Buffering: no`, which nginx honours; other proxies need their own
setting. The symptom is a page that hangs and then delivers everything at once.

**Request timeout.** A question takes six to ten seconds. A default 30s gateway
timeout is fine; a 5s one silently truncates every answer.

## Cost

Measured, not estimated — see `scripts/demo_tracing.py`:

    answering              ~$0.005 per question
    answering + judging    ~$0.015 per evaluated question

The eval suite runs nightly rather than per commit for that reason. CI is split
accordingly in `.github/workflows/checks.yml`.
