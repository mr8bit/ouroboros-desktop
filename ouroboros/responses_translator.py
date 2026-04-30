"""OpenResponses ⇄ Ouroboros bridge translation.

Two responsibilities, deliberately kept in one module so the wire shape and
its inverse live next to each other:

1. ``translate_input_to_user_message`` — turns an OpenResponses request body
   (string or array of input items) into the plain-text user message and
   structured attachments (resolved file/image content) that the
   ``LocalChatBridge`` understands.

2. ``build_response_object`` / ``sse_*`` helpers — render the canonical
   non-streaming response JSON and the streaming SSE event stream from the
   final agent text plus an ordered list of intercepted tool calls.

This module has zero coupling to ``LocalChatBridge`` and zero coupling to
HTTP — it operates on plain dicts and is fully unit-testable.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ouroboros.contracts.openresponses_schema import (
    EVENT_COMPLETED,
    EVENT_CONTENT_PART_ADDED,
    EVENT_CONTENT_PART_DONE,
    EVENT_CREATED,
    EVENT_FAILED,
    EVENT_FUNCTION_ARGS_DELTA,
    EVENT_FUNCTION_ARGS_DONE,
    EVENT_IN_PROGRESS,
    EVENT_OUTPUT_ITEM_ADDED,
    EVENT_OUTPUT_ITEM_DONE,
    EVENT_OUTPUT_TEXT_DELTA,
    EVENT_OUTPUT_TEXT_DONE,
)


# ---------------------------------------------------------------------------
# Untrusted-content envelope (OpenClaw-compatible)
# ---------------------------------------------------------------------------

UNTRUSTED_OPEN_TEMPLATE = '<<<EXTERNAL_UNTRUSTED_CONTENT id="{id}">>>'
UNTRUSTED_CLOSE_TEMPLATE = '<<<END id="{id}">>>'


def wrap_untrusted(body: str, ext_id: Optional[str] = None) -> str:
    """Wrap external file/URL content with the OpenClaw injection-defense markers."""
    eid = ext_id or f"ext_{uuid.uuid4().hex[:12]}"
    return f"{UNTRUSTED_OPEN_TEMPLATE.format(id=eid)}\n{body}\n{UNTRUSTED_CLOSE_TEMPLATE.format(id=eid)}"


# ---------------------------------------------------------------------------
# Resolved attachments (filled by responses_files; carried through executor)
# ---------------------------------------------------------------------------


@dataclass
class ResolvedAttachment:
    """Result of fetching one input_image / input_file item.

    ``kind == 'text'`` means the body should be inlined into the user message
    (already wrapped in EXTERNAL_UNTRUSTED_CONTENT markers).  ``kind == 'image'``
    means the data is a base64-encoded image to be passed alongside the user
    message via the bridge's image_data channel.
    """

    kind: str  # "text" | "image"
    body: str = ""              # text: full content; image: base64 (no data: prefix)
    media_type: str = ""        # image MIME for kind=image
    filename: str = ""          # for display only


@dataclass
class TranslatedInput:
    user_text: str
    system_prefix: str
    attachments: List[ResolvedAttachment] = field(default_factory=list)
    function_call_outputs: List[Tuple[str, str]] = field(default_factory=list)
    """List of (call_id, output) for resume protocol."""


def translate_input_to_user_message(
    body: Union[str, List[Dict[str, Any]]],
    *,
    instructions: str = "",
    resolve_attachments=None,
) -> TranslatedInput:
    """Convert request.input + request.instructions into bridge-compatible parts.

    Parameters
    ----------
    body
        Either a plain string (treated as a single user message) or a list of
        input items per the OpenAI Responses spec (`message`, `input_image`,
        `input_file`, `function_call_output`).
    instructions
        Optional `instructions` field — injected into the system prefix.
    resolve_attachments
        Callable invoked for every input_image / input_file item.  Signature:
        ``resolve(item: dict) -> ResolvedAttachment``.  May raise to abort the
        request with a 400 error (caller wraps the exception).  If ``None``,
        such items are skipped silently — useful for unit tests of the
        text-only path.

    Returns
    -------
    TranslatedInput
        ``user_text`` is what becomes the user message in the bridge inbox.
        ``system_prefix`` is text that should be prepended to the system
        prompt (carries instructions + inlined untrusted file content).
        ``attachments`` are images that need to ride alongside as base64.
        ``function_call_outputs`` carry resume-protocol results.
    """
    user_chunks: List[str] = []
    system_chunks: List[str] = []
    attachments: List[ResolvedAttachment] = []
    fco: List[Tuple[str, str]] = []

    if instructions:
        system_chunks.append(str(instructions).strip())

    if isinstance(body, str):
        user_chunks.append(body)
    elif isinstance(body, list):
        for raw in body:
            if not isinstance(raw, dict):
                continue
            item_type = str(raw.get("type") or "").strip()
            if item_type == "message" or (item_type == "" and "role" in raw):
                role = str(raw.get("role") or "user").strip().lower()
                content = raw.get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = _join_message_content_parts(content)
                else:
                    text = ""
                if not text:
                    continue
                if role in ("user", ""):
                    user_chunks.append(text)
                elif role in ("system", "developer", "assistant"):
                    system_chunks.append(f"[{role}] {text}")
            elif item_type == "input_image":
                if resolve_attachments is None:
                    continue
                resolved = resolve_attachments(raw)
                if resolved is None:
                    continue
                if resolved.kind == "image":
                    attachments.append(resolved)
                elif resolved.kind == "text":
                    system_chunks.append(resolved.body)
            elif item_type == "input_file":
                if resolve_attachments is None:
                    continue
                resolved = resolve_attachments(raw)
                if resolved is None:
                    continue
                if resolved.kind == "image":
                    attachments.append(resolved)
                elif resolved.kind == "text":
                    system_chunks.append(resolved.body)
            elif item_type == "function_call_output":
                call_id = str(raw.get("call_id") or "").strip()
                output = raw.get("output")
                if isinstance(output, (dict, list)):
                    output_str = json.dumps(output, ensure_ascii=False)
                else:
                    output_str = str(output or "")
                if call_id:
                    fco.append((call_id, output_str))
            # reasoning, item_reference, etc. are ignored per OpenClaw spec

    user_text = "\n\n".join(c.strip() for c in user_chunks if c and c.strip())
    system_prefix = "\n\n".join(c.strip() for c in system_chunks if c and c.strip())
    return TranslatedInput(
        user_text=user_text,
        system_prefix=system_prefix,
        attachments=attachments,
        function_call_outputs=fco,
    )


def _join_message_content_parts(parts: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        ptype = str(part.get("type") or "").strip()
        if ptype in ("input_text", "output_text", "text"):
            text = str(part.get("text") or "").strip()
            if text:
                out.append(text)
        # input_image / input_file embedded inside a message.content array are
        # not handled here — they should appear at the top input array level.
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Tool-call records (filled while the agent runs; rendered back to client)
# ---------------------------------------------------------------------------


@dataclass
class CapturedToolCall:
    item_id: str
    call_id: str
    name: str
    arguments_json: str
    result_text: str = ""
    is_error: bool = False


# ---------------------------------------------------------------------------
# Non-streaming response object construction
# ---------------------------------------------------------------------------


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def new_function_call_item_id() -> str:
    return f"fc_{uuid.uuid4().hex[:24]}"


def new_function_call_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def new_tool_result_item_id() -> str:
    return f"tr_{uuid.uuid4().hex[:24]}"


def build_response_object(
    *,
    response_id: str,
    model: str,
    final_text: str,
    tool_calls: List[CapturedToolCall],
    status: str = "completed",
    previous_response_id: Optional[str] = None,
    user_field: Optional[str] = None,
    usage_input_tokens: int = 0,
    usage_output_tokens: int = 0,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical non-streaming response JSON object.

    Order of items in ``output``:
    1. function_call + tool_result pairs in the order they were invoked.
    2. Final assistant message (if ``final_text`` is non-empty).
    """
    output: List[Dict[str, Any]] = []
    for tc in tool_calls:
        output.append({
            "type": "function_call",
            "id": tc.item_id,
            "call_id": tc.call_id,
            "name": tc.name,
            "arguments": tc.arguments_json,
            "status": "completed",
        })
        if tc.result_text:
            output.append({
                "type": "tool_result",
                "id": new_tool_result_item_id(),
                "call_id": tc.call_id,
                "output": tc.result_text,
                "is_error": tc.is_error,
            })

    if final_text:
        output.append({
            "type": "message",
            "id": new_message_id(),
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": final_text,
                "annotations": [],
            }],
        })

    body: Dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "output": output,
        "output_text": final_text,
        "usage": {
            "input_tokens": int(usage_input_tokens or 0),
            "output_tokens": int(usage_output_tokens or 0),
            "total_tokens": int(usage_input_tokens or 0) + int(usage_output_tokens or 0),
        },
    }
    if previous_response_id:
        body["previous_response_id"] = previous_response_id
    if user_field:
        body["user"] = user_field
    if error:
        body["error"] = error
    return body


# ---------------------------------------------------------------------------
# Streaming SSE rendering
# ---------------------------------------------------------------------------


def sse_format(event: str, data: Dict[str, Any]) -> str:
    """Format one SSE event (named, JSON-encoded data, blank-line terminated)."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"


def sse_initial_events(
    *,
    response_id: str,
    model: str,
    previous_response_id: Optional[str] = None,
    user_field: Optional[str] = None,
) -> Iterator[str]:
    """Emit response.created → response.in_progress."""
    base = {
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": model,
            "output": [],
            "output_text": "",
        },
    }
    if previous_response_id:
        base["response"]["previous_response_id"] = previous_response_id
    if user_field:
        base["response"]["user"] = user_field
    yield sse_format(EVENT_CREATED, base)
    yield sse_format(EVENT_IN_PROGRESS, base)


def sse_function_call(
    tool_call: CapturedToolCall,
    output_index: int,
) -> Iterator[str]:
    """Emit the standard function_call event sequence for one tool call.

    Sequence: output_item.added → function_call_arguments.delta →
    function_call_arguments.done → output_item.done.
    """
    item_added = {
        "type": "function_call",
        "id": tool_call.item_id,
        "call_id": tool_call.call_id,
        "name": tool_call.name,
        "arguments": "",
        "status": "in_progress",
    }
    yield sse_format(EVENT_OUTPUT_ITEM_ADDED, {
        "output_index": output_index,
        "item": item_added,
    })
    if tool_call.arguments_json:
        yield sse_format(EVENT_FUNCTION_ARGS_DELTA, {
            "item_id": tool_call.item_id,
            "output_index": output_index,
            "delta": tool_call.arguments_json,
        })
    yield sse_format(EVENT_FUNCTION_ARGS_DONE, {
        "item_id": tool_call.item_id,
        "output_index": output_index,
        "arguments": tool_call.arguments_json,
    })
    item_done = dict(item_added)
    item_done["arguments"] = tool_call.arguments_json
    item_done["status"] = "completed"
    yield sse_format(EVENT_OUTPUT_ITEM_DONE, {
        "output_index": output_index,
        "item": item_done,
    })


def sse_tool_result(tool_call: CapturedToolCall, output_index: int) -> Iterator[str]:
    """Emit a custom tool_result item (OpenClaw extension)."""
    if not tool_call.result_text:
        return
    item = {
        "type": "tool_result",
        "id": new_tool_result_item_id(),
        "call_id": tool_call.call_id,
        "output": tool_call.result_text,
        "is_error": tool_call.is_error,
    }
    yield sse_format(EVENT_OUTPUT_ITEM_ADDED, {
        "output_index": output_index,
        "item": item,
    })
    yield sse_format(EVENT_OUTPUT_ITEM_DONE, {
        "output_index": output_index,
        "item": item,
    })


def sse_message_text(
    text: str,
    output_index: int,
    *,
    chunk_size: int = 1024,
) -> Iterator[str]:
    """Emit the message → content_part → text deltas → done sequence.

    The Ouroboros agent currently posts complete messages rather than streaming
    tokens, so we synthesize text deltas by chunking the final string. This
    preserves the canonical event order so OpenAI clients work unmodified.
    """
    message_id = new_message_id()
    item_added = {
        "type": "message",
        "id": message_id,
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    yield sse_format(EVENT_OUTPUT_ITEM_ADDED, {
        "output_index": output_index,
        "item": item_added,
    })
    yield sse_format(EVENT_CONTENT_PART_ADDED, {
        "item_id": message_id,
        "output_index": output_index,
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    })
    if text:
        for chunk in _chunk_text(text, chunk_size):
            yield sse_format(EVENT_OUTPUT_TEXT_DELTA, {
                "item_id": message_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": chunk,
            })
    yield sse_format(EVENT_OUTPUT_TEXT_DONE, {
        "item_id": message_id,
        "output_index": output_index,
        "content_index": 0,
        "text": text,
    })
    yield sse_format(EVENT_CONTENT_PART_DONE, {
        "item_id": message_id,
        "output_index": output_index,
        "content_index": 0,
        "part": {"type": "output_text", "text": text, "annotations": []},
    })
    item_done = {
        "type": "message",
        "id": message_id,
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    yield sse_format(EVENT_OUTPUT_ITEM_DONE, {
        "output_index": output_index,
        "item": item_done,
    })


def sse_completed(response_obj: Dict[str, Any]) -> str:
    return sse_format(EVENT_COMPLETED, {"response": response_obj})


def sse_failed(error_body: Dict[str, Any]) -> str:
    return sse_format(EVENT_FAILED, {"error": error_body})


def _chunk_text(text: str, chunk_size: int) -> Iterator[str]:
    if not text:
        return
    if chunk_size <= 0 or len(text) <= chunk_size:
        yield text
        return
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
