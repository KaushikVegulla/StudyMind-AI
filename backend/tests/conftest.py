"""Suite-wide test environment.

Set before any test module imports the app:

- ``EMBEDDINGS_BACKEND=none`` — the suite must never download an
  embedding model; retrieval tests exercise the keyword path or mock
  the embedder (same behaviour CI relies on).
- ``RATE_LIMIT_PER_MINUTE`` — raised far above anything the suite
  sends from its single test client IP, so the production rate
  limiter (default 60/min on chat/generate) cannot trip unrelated
  tests. The limiter itself is tested separately with its own app.
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

os.environ.setdefault("EMBEDDINGS_BACKEND", "none")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "100000")
