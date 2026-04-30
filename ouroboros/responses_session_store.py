"""File-backed session store for the OpenResponses gateway.

Tracks the mapping ``response_id → virtual_chat_id`` so follow-up requests
that carry ``previous_response_id`` resume the same agent context.  Also
records the optional ``user`` / ``x-openclaw-session-key`` seed used to
derive deterministic chat ids and the ``paused_for_tool_call`` flag for
the client-tool resume protocol (phase 7).

Storage layout: one JSON file per response_id under
``<DATA_DIR>/responses_sessions/<safe_id>.json``.  Atomic writes (tmp +
rename), TTL-based cleanup, and path-component sanitisation mirror
``ouroboros.a2a_task_store.FileTaskStore`` so the same disk-safety
properties apply.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

log = logging.getLogger("responses-server")


def _safe_id(raw: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", raw or "")
    safe = pathlib.PurePosixPath(safe).name or "invalid_id"
    if not safe.strip("."):
        safe = "invalid_id"
    return safe


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FileResponsesSessionStore:
    """Atomic JSON-file session store for /v1/responses sessions."""

    def __init__(self, data_dir: pathlib.Path, ttl_hours: int = 24):
        self._dir = pathlib.Path(data_dir) / "responses_sessions"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl_hours = max(1, int(ttl_hours))

    def _path(self, response_id: str) -> pathlib.Path:
        return self._dir / f"{_safe_id(response_id)}.json"

    def lookup(self, response_id: str) -> Optional[Dict[str, Any]]:
        if not response_id:
            return None
        path = self._path(response_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Failed to read session %s", response_id, exc_info=True)
            return None

    def upsert(
        self,
        *,
        response_id: str,
        virtual_chat_id: int,
        user_field: Optional[str] = None,
        session_key: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        paused_for_tool_call: bool = False,
        pending_call_id: Optional[str] = None,
    ) -> None:
        existing = self.lookup(response_id) or {}
        record = {
            "response_id": response_id,
            "virtual_chat_id": int(virtual_chat_id),
            "user": user_field or existing.get("user") or None,
            "session_key": session_key or existing.get("session_key") or None,
            "created_at": existing.get("created_at") or _now_iso(),
            "last_used_at": _now_iso(),
            "previous_response_id": previous_response_id or existing.get("previous_response_id") or None,
            "paused_for_tool_call": bool(paused_for_tool_call),
            "pending_call_id": pending_call_id or existing.get("pending_call_id") or None,
        }
        self._atomic_write(self._path(response_id), record)

    def mark_paused(self, response_id: str, pending_call_id: str) -> None:
        record = self.lookup(response_id) or {}
        if not record:
            return
        record["paused_for_tool_call"] = True
        record["pending_call_id"] = pending_call_id
        record["last_used_at"] = _now_iso()
        self._atomic_write(self._path(response_id), record)

    def mark_resumed(self, response_id: str) -> None:
        record = self.lookup(response_id) or {}
        if not record:
            return
        record["paused_for_tool_call"] = False
        record["pending_call_id"] = None
        record["last_used_at"] = _now_iso()
        self._atomic_write(self._path(response_id), record)

    def delete(self, response_id: str) -> None:
        try:
            self._path(response_id).unlink(missing_ok=True)
        except Exception:
            log.debug("Session delete failed for %s", response_id, exc_info=True)

    def cleanup_expired(self) -> int:
        cutoff = time.time() - self._ttl_hours * 3600
        removed = 0
        try:
            for path in self._dir.glob("*.json"):
                try:
                    if path.stat().st_mtime > cutoff:
                        continue
                    path.unlink(missing_ok=True)
                    removed += 1
                except Exception:
                    continue
        except Exception:
            log.warning("Session cleanup error", exc_info=True)
        return removed

    @staticmethod
    def _atomic_write(path: pathlib.Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(record, ensure_ascii=False, indent=2)
        tmp = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex[:8]}")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, content.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(str(tmp), str(path))
