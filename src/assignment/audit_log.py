"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input and start time; return the request ID for correlation."""
        request_id = request_id or uuid4().hex
        self._open[request_id] = {
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "start_time": time.perf_counter(),
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store response, decision, and elapsed time for a request."""
        started = self._open.pop(request_id, None) if request_id else None
        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "input": started["input"] if started else None,
            "started_at": started["started_at"] if started else None,
            "completed_at": utc_now_iso(),
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round((time.perf_counter() - started["start_time"]) * 1000, 2)
            if started else None,
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
