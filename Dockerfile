# Multi-stage: node builds the UI, python runs the service. The final image
# carries no node_modules, no npm and no source — about 180MB instead of 1.2GB,
# and a smaller attack surface because a build toolchain that is not present
# cannot be exploited.

# ---------------------------------------------------------------- UI build
FROM node:22-slim AS ui

WORKDIR /ui
# package files first: this layer is cached until dependencies actually change,
# so editing a component does not reinstall node_modules.
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
# Vite is configured to emit into ../app/api/static, so redirect it to a path
# that exists in this stage.
RUN npm run build -- --outDir dist --emptyOutDir

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

# Fail fast and log immediately. Without PYTHONUNBUFFERED a crash in a container
# can lose the traceback that explains it.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
# docs/ carries the regression baseline the service does not need, but the
# eval scripts do when run inside the image. scripts/ is deliberately NOT
# copied: nothing the service imports lives there any more.
COPY docs ./docs
RUN pip install --no-cache-dir .

COPY --from=ui /ui/dist ./app/api/static

# Non-root. A container that runs as root turns a path-traversal bug into a
# host compromise.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Checks that the service can actually answer, not merely that the port is open.
# A process accepting connections while misconfigured is the failure an
# orchestrator should see.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
b=json.load(urllib.request.urlopen('http://localhost:8000/health')); \
sys.exit(0 if b['chunks_indexed']>0 else 1)"

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
