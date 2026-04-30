"""Bridge between an /v1/responses HTTP request and the Ouroboros agent.

The executor is HTTP-agnostic — it accepts a parsed ``CreateResponseRequest``
plus a ``response_id`` and yields either a single response object
(non-streaming) or an asynchronous iterator of SSE strings (streaming).

The Ouroboros agent posts whole messages (not token streams) via the
``LocalChatBridge``, so for streaming we synthesize text deltas by chunking
the final assistant message — the canonical OpenAI event order is preserved
so any OpenAI-shape client works unmodified.

Tool-call visibility (phase 4) plugs in via the optional
``ToolCallCapture`` parameter — when supplied, the executor will subscribe to
the bridge for tool_call_started / tool_call_finished log events for the
duration of the request and surface them as ``function_call`` and
``tool_result`` items.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from ouroboros.responses_translator import (
    CapturedToolCall,
    TranslatedInput,
    build_response_object,
    new_function_call_item_id,
    new_function_call_call_id,
    sse_completed,
    sse_done,
    sse_failed,
    sse_function_call,
    sse_initial_events,
    sse_message_text,
    sse_tool_result,
    translate_input_to_user_message,
)

log = logging.getLogger("responses-server")

# Virtual chat_id range for /v1/responses traffic.
# A2A uses negative ids descending from -1000; we pick a non-overlapping
# block well below that to make accidental collisions impossible even with
# millions of A2A tasks.
_RESPONSES_CHAT_ID_BASE = -1_000_000_000
_responses_seq = 0
_responses_seq_lock = threading.Lock()


def next_responses_chat_id() -> int:
    """Allocate a fresh virtual chat_id from the responses range."""
    global _responses_seq
    with _responses_seq_lock:
        _responses_seq += 1
        return _RESPONSES_CHAT_ID_BASE - _responses_seq


def stable_responses_chat_id(seed: str) -> int:
    """Deterministic virtual chat_id from a seed (session-key or user field).

    Hashes ``seed`` into the responses negative range so the same seed
    repeatedly resolves to the same agent context.  Range is 1B values wide
    (more than enough for every plausible session-key cardinality).
    """
    import hashlib
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:7], "big") % 900_000_000  # leave headroom
    return _RESPONSES_CHAT_ID_BASE - 1 - offset


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class _ConcurrencyGate:
    """Bounded concurrent execution gate (similar to A2A's semaphore)."""

    def __init__(self, max_concurrent: int):
        self._sem = threading.Semaphore(max(1, int(max_concurrent)))

    def try_acquire(self) -> bool:
        return self._sem.acquire(blocking=False)

    def release(self) -> None:
        self._sem.release()


_GATE: Optional[_ConcurrencyGate] = None
_GATE_LOCK = threading.Lock()


def configure_concurrency(max_concurrent: int) -> None:
    global _GATE
    with _GATE_LOCK:
        _GATE = _ConcurrencyGate(max_concurrent)


def _gate() -> _ConcurrencyGate:
    global _GATE
    with _GATE_LOCK:
        if _GATE is None:
            _GATE = _ConcurrencyGate(int(os.environ.get("OUROBOROS_RESPONSES_MAX_CONCURRENT", "3") or "3"))
        return _GATE


# ---------------------------------------------------------------------------
# Tool-call capture (phase 4 plug)
# ---------------------------------------------------------------------------


class ToolCallCapture:
    """Collects tool_call_* events from the bridge for one in-flight request.

    Phase 1-3 wire this in but the actual subscription is added in phase 4
    (requires extending ``LocalChatBridge`` with a chat-event subscription
    primitive).  For now ``calls`` stays empty so the existing event order
    holds.
    """

    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self.calls: List[CapturedToolCall] = []
        self._call_index: Dict[str, CapturedToolCall] = {}
        self._lock = threading.Lock()

    def on_tool_started(self, tool_call_id: str, name: str, arguments: Any) -> None:
        import json
        with self._lock:
            args = arguments if isinstance(arguments, str) else json.dumps(
                arguments or {}, ensure_ascii=False, default=str,
            )
            entry = CapturedToolCall(
                item_id=new_function_call_item_id(),
                call_id=tool_call_id or new_function_call_call_id(),
                name=name or "",
                arguments_json=args or "{}",
            )
            self._call_index[tool_call_id or entry.call_id] = entry
            self.calls.append(entry)

    def on_tool_finished(self, tool_call_id: str, result_preview: str, is_error: bool) -> None:
        with self._lock:
            entry = self._call_index.get(tool_call_id)
            if entry is None:
                return
            entry.result_text = str(result_preview or "")
            entry.is_error = bool(is_error)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def resolve_chat_id(
    *,
    previous_response_id: Optional[str],
    session_key_header: Optional[str],
    user_field: Optional[str],
    session_store=None,
) -> Tuple[int, bool]:
    """Pick a virtual chat_id.

    Returns ``(chat_id, is_resumed)`` — ``is_resumed`` is True if the chat
    was looked up from a previous_response_id (and therefore the agent
    history continues).
    """
    if previous_response_id and session_store is not None:
        record = session_store.lookup(previous_response_id)
        if record is not None:
            return int(record["virtual_chat_id"]), True
    if session_key_header:
        return stable_responses_chat_id(f"key:{session_key_header.strip()}"), False
    if user_field:
        return stable_responses_chat_id(f"user:{user_field.strip()}"), False
    return next_responses_chat_id(), False


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------


class ConcurrencyRejected(RuntimeError):
    """Raised when no slots are available."""


class BridgeUnavailable(RuntimeError):
    """Raised when the supervisor bridge has not been initialized yet."""


def _coerce_image_data(translated: TranslatedInput) -> Optional[Tuple[str, str, str]]:
    """Pick the first image attachment for handle_chat_direct(image_data=).

    handle_chat_direct accepts only one image at a time. If multiple images
    are supplied, the first wins; the rest are dropped (and we log it).
    """
    images = [a for a in translated.attachments if a.kind == "image" and a.body]
    if not images:
        return None
    if len(images) > 1:
        log.info("Responses request had %d images; only the first will be attached", len(images))
    head = images[0]
    return (head.body, head.media_type or "image/png", head.filename or "")


def _compose_user_text(translated: TranslatedInput) -> str:
    """Combine system prefix and user text into a single bridge message.

    The bridge chat path does not have a separate system slot, so we prepend
    the system prefix as a clearly-marked block.  ``EXTERNAL_UNTRUSTED_CONTENT``
    wrapping is already applied by ``responses_files``.
    """
    if not translated.system_prefix and not translated.function_call_outputs:
        return translated.user_text
    parts: List[str] = []
    if translated.system_prefix:
        parts.append(translated.system_prefix)
    if translated.function_call_outputs:
        for call_id, output in translated.function_call_outputs:
            parts.append(
                f"[function_call_output call_id={call_id}]\n{output}"
            )
    if translated.user_text:
        parts.append(translated.user_text)
    return "\n\n".join(parts)


async def execute_non_streaming(
    *,
    response_id: str,
    model: str,
    chat_id: int,
    translated: TranslatedInput,
    capture: Optional[ToolCallCapture] = None,
    previous_response_id: Optional[str] = None,
    user_field: Optional[str] = None,
    timeout_sec: Optional[int] = None,
) -> Dict[str, Any]:
    """Submit to the agent, wait for completion, return a response object."""
    final_text = await _run_through_bridge(
        chat_id=chat_id,
        translated=translated,
        timeout_sec=timeout_sec,
        capture=capture,
    )
    tool_calls = list(capture.calls) if capture else []
    return build_response_object(
        response_id=response_id,
        model=model,
        final_text=final_text,
        tool_calls=tool_calls,
        previous_response_id=previous_response_id,
        user_field=user_field,
    )


async def execute_streaming(
    *,
    response_id: str,
    model: str,
    chat_id: int,
    translated: TranslatedInput,
    capture: Optional[ToolCallCapture] = None,
    previous_response_id: Optional[str] = None,
    user_field: Optional[str] = None,
    timeout_sec: Optional[int] = None,
) -> AsyncIterator[str]:
    """Yield SSE strings end-to-end for one /v1/responses request."""
    # 1. Initial frames before the agent has produced anything.
    for frame in sse_initial_events(
        response_id=response_id,
        model=model,
        previous_response_id=previous_response_id,
        user_field=user_field,
    ):
        yield frame

    try:
        final_text = await _run_through_bridge(
            chat_id=chat_id,
            translated=translated,
            timeout_sec=timeout_sec,
            capture=capture,
        )
    except asyncio.TimeoutError as exc:
        yield sse_failed({
            "message": str(exc) or "Agent did not respond in time",
            "type": "server_error",
        })
        yield sse_done()
        return
    except BridgeUnavailable as exc:
        yield sse_failed({"message": str(exc), "type": "server_error"})
        yield sse_done()
        return
    except Exception as exc:  # pragma: no cover — defensive
        log.error("responses streaming dispatch failed: %s", exc, exc_info=True)
        yield sse_failed({"message": "internal error", "type": "server_error"})
        yield sse_done()
        return

    output_index = 0
    captured = list(capture.calls) if capture else []
    for tc in captured:
        for frame in sse_function_call(tc, output_index):
            yield frame
        output_index += 1
        if tc.result_text:
            for frame in sse_tool_result(tc, output_index):
                yield frame
            output_index += 1

    if final_text:
        for frame in sse_message_text(final_text, output_index):
            yield frame
        output_index += 1

    final_obj = build_response_object(
        response_id=response_id,
        model=model,
        final_text=final_text,
        tool_calls=captured,
        previous_response_id=previous_response_id,
        user_field=user_field,
    )
    yield sse_completed(final_obj)
    yield sse_done()


# ---------------------------------------------------------------------------
# Bridge plumbing
# ---------------------------------------------------------------------------


async def _run_through_bridge(
    *,
    chat_id: int,
    translated: TranslatedInput,
    timeout_sec: Optional[int],
    capture: Optional["ToolCallCapture"] = None,
) -> str:
    """Inject one user message and wait for the agent's final reply.

    Mirrors the A2A executor pattern: subscribe to the bridge for our
    chat_id, dispatch via ``handle_chat_direct`` in a thread, await the
    response_event with a hard timeout.

    When ``capture`` is provided, additionally subscribes to per-chat-id
    bridge events for the duration of the request so tool_call_started /
    tool_call_finished events are recorded.
    """
    from supervisor.message_bus import try_get_bridge
    from supervisor.workers import handle_chat_direct

    bridge = try_get_bridge()
    if bridge is None:
        raise BridgeUnavailable("Supervisor not ready — bridge not initialized yet")

    user_text = _compose_user_text(translated)
    if not user_text and not translated.attachments:
        return "(empty input)"

    image_data = _coerce_image_data(translated)
    response_event = asyncio.Event()
    response_holder: Dict[str, str] = {}
    loop = asyncio.get_running_loop()

    def on_response(text: str) -> None:
        response_holder["text"] = text
        loop.call_soon_threadsafe(response_event.set)

    def on_chat_event(payload: Dict[str, Any]) -> None:
        if capture is None:
            return
        if not isinstance(payload, dict):
            return
        if payload.get("type") != "log":
            return
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            return
        ev_type = str(data.get("type") or "")
        if ev_type == "tool_call_started":
            capture.on_tool_started(
                tool_call_id=str(data.get("tool_call_id") or data.get("id") or ""),
                name=str(data.get("tool") or ""),
                arguments=data.get("args"),
            )
        elif ev_type == "tool_call_finished":
            capture.on_tool_finished(
                tool_call_id=str(data.get("tool_call_id") or data.get("id") or ""),
                result_preview=str(data.get("result_preview") or ""),
                is_error=bool(data.get("is_error") or False),
            )

    sub_response = bridge.subscribe_response(chat_id, on_response)
    sub_events = bridge.subscribe_chat_events(chat_id, on_chat_event)
    timeout = timeout_sec or _resolve_timeout()
    try:
        await asyncio.to_thread(handle_chat_direct, chat_id, user_text, image_data)
        try:
            await asyncio.wait_for(response_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Agent did not respond within {timeout}s")
    finally:
        bridge.unsubscribe_response(sub_response)
        bridge.unsubscribe_chat_events(sub_events)
    return response_holder.get("text", "(no response)")


def _resolve_timeout() -> int:
    raw = os.environ.get("OUROBOROS_HARD_TIMEOUT_SEC", "1800")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 1800


# ---------------------------------------------------------------------------
# Public single-shot helper used by the HTTP handler
# ---------------------------------------------------------------------------


def acquire_slot() -> bool:
    """Try to take a concurrency slot.  Caller must call release_slot()."""
    return _gate().try_acquire()


def release_slot() -> None:
    _gate().release()


def make_translated_input(
    request: Dict[str, Any],
    *,
    resolve_attachments=None,
) -> TranslatedInput:
    """Convenience: pull out input + instructions and translate them."""
    return translate_input_to_user_message(
        request.get("input") or [],
        instructions=str(request.get("instructions") or ""),
        resolve_attachments=resolve_attachments,
    )
