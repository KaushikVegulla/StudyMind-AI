"""Supabase / pgvector connection placeholder.

Returns None until SUPABASE_URL and SUPABASE_SERVICE_KEY are set, so the
scaffold runs in-memory. The schema lives in supabase/schema.sql.
"""
from __future__ import annotations

import os


def get_client():
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    from supabase import create_client  # optional dependency, later step

    return create_client(url, key)
