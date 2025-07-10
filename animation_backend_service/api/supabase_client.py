"""Supabase client singleton and helper utilities.

This module centralises all interactions with Supabase so that the rest of the
backend (API routes + Celery tasks) can import a single place.  We use the
Service key so server-side code has full RLS bypass permissions.
"""

from __future__ import annotations

import io
import uuid
from typing import Optional
from datetime import datetime

from supabase import create_client, Client

from .config import settings


# ---------------------------------------------------------------------------
# Client initialisation (singleton pattern)
# ---------------------------------------------------------------------------

_SUPABASE_CLIENT: Optional[Client] = None


def get_supabase() -> Client:
    """Return a lazily-initialised Supabase client using service secret key."""
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is None:
        _SUPABASE_CLIENT = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY,
        )
    return _SUPABASE_CLIENT


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def upload_bytes_to_bucket(
    *,
    bucket: str,
    content: bytes,
    mime_type: str = "application/octet-stream",
    user_scope: str = "default",
    extension: str = "bin",
) -> str:
    """Upload raw bytes to a bucket and return the public URL.

    Args:
        bucket:   Storage bucket name.
        content:  Raw file bytes.
        mime_type:Content-Type header to set.
        user_scope:A folder/prefix to group files (e.g., user id).
        extension:File extension for naming.
    """
    supabase = get_supabase()

    filename = f"{user_scope}/{uuid.uuid4()}.{extension}"
    # Upload (upsert=True so retries overwrite)
    options = {"contentType": mime_type}
    res = supabase.storage.from_(bucket).upload(filename, content, options)

    # The SDK may return a dict with 'error' or a `requests.Response`.
    if isinstance(res, dict):
        if res.get("error") is not None:
            raise RuntimeError(f"Supabase upload failed: {res['error']}")
    else:
        # Assume `res` is an HTTPX/requests Response-like object
        if res.status_code >= 400:
            raise RuntimeError(
                f"Supabase upload failed with status {res.status_code}: {getattr(res, 'text', '')}"
            )

    public_res = supabase.storage.from_(bucket).get_public_url(filename)

    public: str | None = None
    if isinstance(public_res, dict):
        # Newer SDK (<2) returns dict
        public = (
            public_res.get("data", {}).get("publicUrl")
            if isinstance(public_res.get("data"), dict)
            else public_res.get("publicUrl")
        )
    elif hasattr(public_res, "data"):
        # Older SDK object with .data attr (dict)
        public = public_res.data["publicUrl"]
    elif isinstance(public_res, str):
        public = public_res

    if not public:
        raise RuntimeError("Could not retrieve public URL from Supabase response")
    return public


def upload_fileobj_to_bucket(*, bucket: str, file_obj, user_scope: str, filename: str) -> str:
    """Upload a Python file-like object (opened in binary mode) to Storage."""
    data = file_obj.read()
    ext = filename.split(".")[-1]
    mime = "image/png" if ext in {"png"} else "image/jpeg"
    return upload_bytes_to_bucket(
        bucket=bucket,
        content=data,
        mime_type=mime,
        user_scope=user_scope,
        extension=ext,
    )


# ---------------------------------------------------------------------------
# Database helpers (uploaded_files & timeline_slots)
# ---------------------------------------------------------------------------


def insert_uploaded_file(*, scene_id: str, original_id: str, name: str, url: str) -> str:
    """Insert row into uploaded_files; returns new file id."""
    supabase = get_supabase()
    res = (
        supabase.table("uploaded_files")
        .insert({
            "scene_id": scene_id,
            "original_id": original_id,
            "name": name,
            "url": url,
        })
        .execute()
    )
    # Check for error by status_code or data
    if hasattr(res, 'status_code') and res.status_code >= 400:
        raise RuntimeError(f"insert_uploaded_file failed: {getattr(res, 'text', '')}")
    data = getattr(res, 'data', None)
    if not data or not isinstance(data, list) or not data[0].get("id"):
        raise RuntimeError(f"insert_uploaded_file: No id returned in response: {data}")
    return data[0]["id"]


def insert_timeline_slot(*, project_id: str, slot_index: int, file_id: str, slot_type: str = "generated") -> str:
    supabase = get_supabase()
    payload = {
        "project_id": project_id,
        "slot_index": slot_index,
        "file_id": file_id,
        "type": slot_type,  # now defaults to 'generated'
        "locked": False,
        "updated_at": datetime.utcnow().isoformat(),  # Always update
    }
    print(f"[DEBUG] Upserting timeline slot with payload: {payload}")
    res = (
        supabase.table("timeline_slots")
        .upsert(payload, on_conflict="project_id,slot_index")
        .execute()
    )
    print(f"[DEBUG] Upsert response: status_code={getattr(res, 'status_code', None)}, data={getattr(res, 'data', None)}, text={getattr(res, 'text', None)}")
    if hasattr(res, 'status_code') and res.status_code >= 400:
        raise RuntimeError(f"insert_timeline_slot failed: {getattr(res, 'text', '')}")
    data = getattr(res, 'data', None)
    if not data or not isinstance(data, list) or not data[0].get("id"):
        raise RuntimeError(f"insert_timeline_slot: No id returned in response: {data}")
    return data[0]["id"] 