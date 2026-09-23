#!/bin/sh
# Offline checks before a push. Install with:
#   git config core.hooksPath .githooks
#
# Deliberately excludes the live evals: they cost money and take a minute, and a
# hook that is slow or expensive gets bypassed with --no-verify. A check nobody
# runs protects nothing — which is the same reasoning behind the tolerances in
# app/core/regression.py.
set -e
echo "running offline checks..."
pytest -q
python scripts/check_regressions.py
