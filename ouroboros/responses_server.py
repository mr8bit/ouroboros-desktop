"""OpenResponses HTTP gateway — Ouroboros side.

Runs a separate Starlette/uvicorn server on
``OUROBOROS_RESPONSES_PORT`` (default 18789, matching OpenClaw).  Disabled by
default; toggled via ``OUROBOROS_RESPONSES_ENABLED`` in Settings →
Integrations.  Requires restart when toggled, port-changed, or
host-rebound.

Routes:
    POST /v1/responses    — main endpoint (streaming or non-streaming)
    GET  /healthz          — liveness check (returns {"ok": true})

Authentication: ``Authorization: Bearer <OUROBOROS_RESPONSES_TOKEN>``.  The
token is mandatory — the server refuses to start when the gateway is
enabled but the token is empty (loud failure, mirrors OpenClaw's
shared-secret model).

The HTTP layer here is deliberately thin.  Translation lives in
``responses_translator``; bridge dispatch lives in ``responses_executor``;
file/image fetching lives in ``responses_files``; sessions live in
``responses_session_store``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

try:
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response, StreamingResponse
    from starlette.routing import Route
    _STARLETTE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _STARLETTE_AVAILABLE = False
    uvicorn = None  # type: ignore[assignment]
    Starlette = Request = None  # type: ignore[assignment]
    JSONResponse = Response = StreamingResponse = None  # type: ignore[assignment]
    Route = None  # type: ignore[assignment]

from ouroboros.responses_executor import (
    ToolCallCapture,
    acquire_slot,
    configure_concurrency,
    execute_non_streaming,
    execute_streaming,
    make_translated_input,
    release_slot,
    resolve_chat_id,
)
from ouroboros.responses_translator import new_response_id

log = logging.getLogger("responses-server")

# Module-level state for lifecycle management
_server: Optional["uvicorn.Server"] = None
_cleanup_task: Optional[asyncio.Task] = None
_session_store_ref: Any = None  # set when session store exists (phase 5)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(data_dir: pathlib.Path) -> None:
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "responses.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    if not log.handlers:
        log.addHandler(handler)
        log.addHandler(logging.StreamHandler())
        log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _expected_token() -> str:
    return str(os.environ.get("OUROBOROS_RESPONSES_TOKEN", "") or "").strip()


def _bearer_from(request: "Request") -> str:
    raw = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    raw = raw.strip()
    if raw.lower().startswith("bearer "):
        return raw.split(None, 1)[1].strip()
    return ""


def _auth_failure() -> "JSONResponse":
    return JSONResponse(
        {"error": {"message": "Missing or invalid bearer token", "type": "authentication_error"}},
        status_code=401,
        headers={"WWW-Authenticate": 'Bearer realm="ouroboros-responses"'},
    )


def _check_auth(request: "Request") -> Optional["JSONResponse"]:
    expected = _expected_token()
    if not expected:
        # Server-side misconfiguration — gateway is enabled but no token set.
        # We refuse all requests rather than serve open-by-default.
        log.error("OUROBOROS_RESPONSES_TOKEN is empty — refusing every /v1/responses request")
        return JSONResponse(
            {"error": {"message": "Gateway not configured (missing token)", "type": "server_error"}},
            status_code=503,
        )
    presented = _bearer_from(request)
    if not presented:
        return _auth_failure()
    if not _constant_time_eq(presented, expected):
        return _auth_failure()
    return None


def _constant_time_eq(a: str, b: str) -> bool:
    import hmac
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# ---------------------------------------------------------------------------
# Body parsing
# ---------------------------------------------------------------------------

# Cap the request body size up-front so a hostile client cannot exhaust memory
# by streaming an unbounded JSON blob.  Image/file payloads are validated
# again later inside ``responses_files`` against tighter per-modality limits.
_MAX_BODY_BYTES_DEFAULT = 16 * 1024 * 1024  # 16 MiB
_MAX_BODY_BYTES_ENV = "OUROBOROS_RESPONSES_MAX_BODY_BYTES"


def _max_body_bytes() -> int:
    raw = os.environ.get(_MAX_BODY_BYTES_ENV, "")
    try:
        return int(raw) if raw else _MAX_BODY_BYTES_DEFAULT
    except ValueError:
        return _MAX_BODY_BYTES_DEFAULT


async def _read_body(request: "Request") -> tuple[bytes, Optional["Response"]]:
    cap = _max_body_bytes()
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > cap:
        return b"", JSONResponse(
            {"error": {"message": f"Body exceeds {cap} bytes", "type": "invalid_request_error"}},
            status_code=413,
        )
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        chunks.append(chunk)
        total += len(chunk)
        if total > cap:
            return b"", JSONResponse(
                {"error": {"message": f"Body exceeds {cap} bytes", "type": "invalid_request_error"}},
                status_code=413,
            )
    return b"".join(chunks), None


def _parse_json_body(body: bytes) -> tuple[Optional[Dict[str, Any]], Optional["Response"]]:
    if not body:
        return None, JSONResponse(
            {"error": {"message": "Empty body", "type": "invalid_request_error"}},
            status_code=400,
        )
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, JSONResponse(
            {"error": {"message": f"Invalid JSON: {exc}", "type": "invalid_request_error"}},
            status_code=400,
        )
    if not isinstance(parsed, dict):
        return None, JSONResponse(
            {"error": {"message": "Body must be a JSON object", "type": "invalid_request_error"}},
            status_code=400,
        )
    return parsed, None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_MODEL_PREFIX = "openclaw"


def _validate_request(payload: Dict[str, Any]) -> Optional["Response"]:
    model = str(payload.get("model") or "").strip()
    if not model:
        return JSONResponse(
            {"error": {"message": "model is required", "type": "invalid_request_error", "param": "model"}},
            status_code=400,
        )
    # Accept "openclaw", "openclaw/default", "openclaw/<id>".  Any other value is
    # rejected so misrouted OpenAI requests fail loudly instead of silently
    # going through this single-agent gateway.
    head = model.split("/", 1)[0]
    if head != _VALID_MODEL_PREFIX:
        return JSONResponse(
            {"error": {
                "message": (
                    f"Unsupported model '{model}'. This gateway only serves the "
                    "Ouroboros agent — use 'openclaw', 'openclaw/default', or "
                    "'openclaw/<agent>'."
                ),
                "type": "invalid_request_error",
                "param": "model",
            }},
            status_code=400,
        )
    body = payload.get("input")
    if body is None:
        return JSONResponse(
            {"error": {"message": "input is required", "type": "invalid_request_error", "param": "input"}},
            status_code=400,
        )
    if not isinstance(body, (str, list)):
        return JSONResponse(
            {"error": {"message": "input must be a string or an array", "type": "invalid_request_error", "param": "input"}},
            status_code=400,
        )
    return None


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


async def _create_response(request: "Request") -> "Response":
    auth_err = _check_auth(request)
    if auth_err is not None:
        return auth_err

    body_bytes, body_err = await _read_body(request)
    if body_err is not None:
        return body_err
    payload, parse_err = _parse_json_body(body_bytes)
    if parse_err is not None:
        return parse_err
    assert payload is not None  # for type-checker after parse_err guard

    validation_err = _validate_request(payload)
    if validation_err is not None:
        return validation_err

    if not acquire_slot():
        return JSONResponse(
            {"error": {"message": "Too many concurrent requests", "type": "rate_limit_error"}},
            status_code=429,
        )

    response_id = new_response_id()
    model = str(payload.get("model") or "").strip()
    stream = bool(payload.get("stream") or False)
    previous_response_id = (str(payload.get("previous_response_id") or "").strip()) or None
    user_field = (str(payload.get("user") or "").strip()) or None
    session_key_header = (request.headers.get("x-openclaw-session-key") or "").strip() or None

    # Phase 1-3: file/image resolver is a no-op; phase 6 wires in
    # ``responses_files.build_resolver``.  Until then, file/image input items
    # are silently skipped (translator already handles None gracefully).
    try:
        from ouroboros.responses_files import build_resolver  # type: ignore
        resolver = build_resolver()
    except ImportError:
        resolver = None

    try:
        translated = make_translated_input(payload, resolve_attachments=resolver)
    except ValueError as exc:
        # AttachmentRejected (responses_files) and translator validation
        # errors both subclass ValueError — surface as 400 with the precise
        # rejection reason.
        release_slot()
        return JSONResponse(
            {"error": {"message": str(exc), "type": "invalid_request_error"}},
            status_code=400,
        )
    except Exception as exc:
        release_slot()
        log.error("Failed to translate input: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": {"message": "Failed to process input", "type": "server_error"}},
            status_code=500,
        )

    # Phase 7 (client tools / pause-resume) is intentionally limited in v1:
    # if the client supplies `tools`, the agent cannot route through them
    # because the agent runs to completion synchronously per turn.  Reject
    # with a precise 400 instead of silently ignoring — clients will know to
    # use server-side tools for now.
    client_tools = payload.get("tools") or []
    if client_tools and isinstance(client_tools, list):
        release_slot()
        return JSONResponse(
            {"error": {
                "message": (
                    "Client-provided tools are not yet supported by this gateway. "
                    "The agent uses its own internal tools (read_file, write_file, "
                    "shell, etc.); their calls are surfaced as standard "
                    "function_call output items."
                ),
                "type": "invalid_request_error",
                "param": "tools",
            }},
            status_code=400,
        )

    chat_id, _resumed = resolve_chat_id(
        previous_response_id=previous_response_id,
        session_key_header=session_key_header,
        user_field=user_field,
        session_store=_session_store_ref,
    )
    if _session_store_ref is not None:
        try:
            _session_store_ref.upsert(
                response_id=response_id,
                virtual_chat_id=chat_id,
                user_field=user_field,
                session_key=session_key_header,
                previous_response_id=previous_response_id,
            )
        except Exception:
            log.warning("Session store upsert failed", exc_info=True)

    capture: Optional[ToolCallCapture] = None
    # Tool-call visibility is enabled by phase 4 once the bridge subscription
    # primitive lands.  ``ToolCallCapture`` is constructed regardless so that
    # subscription wiring can flip on by setting one bridge hook.
    capture = ToolCallCapture(chat_id=chat_id)

    if stream:
        async def _gen():
            try:
                async for frame in execute_streaming(
                    response_id=response_id,
                    model=model,
                    chat_id=chat_id,
                    translated=translated,
                    capture=capture,
                    previous_response_id=previous_response_id,
                    user_field=user_field,
                ):
                    yield frame.encode("utf-8")
            finally:
                release_slot()

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx/cloud buffering if present
                "Connection": "keep-alive",
            },
        )

    # Non-streaming
    try:
        try:
            obj = await execute_non_streaming(
                response_id=response_id,
                model=model,
                chat_id=chat_id,
                translated=translated,
                capture=capture,
                previous_response_id=previous_response_id,
                user_field=user_field,
            )
        except asyncio.TimeoutError as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "server_error"}},
                status_code=504,
            )
        except Exception as exc:
            log.error("/v1/responses non-streaming dispatch failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": {"message": "internal error", "type": "server_error"}},
                status_code=500,
            )
        return JSONResponse(obj)
    finally:
        release_slot()


async def _healthz(_request: "Request") -> "Response":
    return JSONResponse({"ok": True, "service": "ouroboros-responses"})


async def _method_not_allowed(_request: "Request") -> "Response":
    return JSONResponse(
        {"error": {"message": "Method not allowed", "type": "invalid_request_error"}},
        status_code=405,
        headers={"Allow": "POST"},
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def _session_cleanup_loop(store, interval_sec: int = 3600) -> None:
    while True:
        await asyncio.sleep(interval_sec)
        try:
            removed = await asyncio.to_thread(store.cleanup_expired)
            if removed:
                log.info("Responses session cleanup: removed %d expired entries", removed)
        except Exception:
            log.warning("Responses session cleanup error", exc_info=True)


def _build_app() -> "Starlette":
    routes = [
        Route("/v1/responses", endpoint=_create_response, methods=["POST"]),
        Route("/v1/responses", endpoint=_method_not_allowed),
        Route("/healthz", endpoint=_healthz, methods=["GET"]),
    ]
    return Starlette(routes=routes)


async def start_responses_server(settings: Dict[str, Any]) -> None:
    """Start the OpenResponses gateway as an async task."""
    global _server, _cleanup_task, _session_store_ref

    if not _STARLETTE_AVAILABLE:
        log.warning("Starlette/uvicorn not available — responses gateway cannot start")
        return

    from ouroboros.config import DATA_DIR

    host = str(settings.get("OUROBOROS_RESPONSES_HOST", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(settings.get("OUROBOROS_RESPONSES_PORT", 18789) or 18789)
    max_concurrent = int(settings.get("OUROBOROS_RESPONSES_MAX_CONCURRENT", 3) or 3)
    ttl_hours = int(settings.get("OUROBOROS_RESPONSES_SESSION_TTL_HOURS", 24) or 24)
    token = str(settings.get("OUROBOROS_RESPONSES_TOKEN", "") or "").strip()

    _setup_logging(DATA_DIR)
    configure_concurrency(max_concurrent)
    os.environ["OUROBOROS_RESPONSES_TOKEN"] = token  # so handler reads the same value

    if not token:
        log.error(
            "OUROBOROS_RESPONSES_ENABLED is true but OUROBOROS_RESPONSES_TOKEN is empty. "
            "Set a strong shared secret in Settings → Integrations and restart. "
            "The gateway will run, but all requests will return 503 until a token is set."
        )

    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "Responses gateway binding to non-loopback host %s — ensure the bearer "
            "token is strong and the port is firewalled appropriately.",
            host,
        )

    log.info("Starting OpenResponses gateway on %s:%d", host, port)

    # Session store (phase 5) — best-effort import; no-op until module exists.
    try:
        from ouroboros.responses_session_store import FileResponsesSessionStore  # type: ignore
        store = FileResponsesSessionStore(DATA_DIR, ttl_hours=ttl_hours)
        _session_store_ref = store
    except ImportError:
        store = None
        _session_store_ref = None

    app = _build_app()

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
    )
    _server = uvicorn.Server(config)

    local_cleanup_task: Optional[asyncio.Task] = None
    if store is not None:
        local_cleanup_task = asyncio.create_task(
            _session_cleanup_loop(store), name="responses-session-cleanup"
        )
        _cleanup_task = local_cleanup_task

    try:
        await _server.serve()
    except Exception:
        log.error("Responses gateway on %s:%d exited with error", host, port, exc_info=True)
    finally:
        if local_cleanup_task is not None and not local_cleanup_task.done():
            local_cleanup_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(local_cleanup_task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass


def stop_responses_server() -> None:
    """Signal the gateway to stop (sync, safe from non-async / panic context)."""
    global _server, _cleanup_task
    if _server is not None:
        _server.should_exit = True
        _server = None
    _cleanup_task = None
    log.info("Responses gateway shutdown requested")
