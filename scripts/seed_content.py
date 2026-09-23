"""Compatibility shim.

The corpus moved to app/rag/corpus.py because the running service imports it.
Kept as a re-export so existing scripts keep working.
"""

from app.rag.corpus import PAGES, SeedPage, load_offline_corpus  # noqa: F401