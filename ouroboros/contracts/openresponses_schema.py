"""OpenResponses HTTP API schema (OpenClaw flavor of OpenAI /v1/responses).

These TypedDicts describe the exact wire envelopes accepted and emitted by
``ouroboros.responses_server``.  They mirror the OpenClaw documentation at
https://docs.openclaw.ai/gateway/openresponses-http-api with a few intentional
deltas, all of which are listed in ``RESPONSES_GATEWAY_DELTAS`` below.

These are descriptive contracts, not validators — nothing rejects extra or
missing fields automatically.  Their job is to:

- make the request/response shape *visible* as a stable surface,
- anchor regression tests,
- give a single home for new envelope keys when the gateway evolves.

Conventions
-----------
- Default to ``total=True`` (top-level keys are required).
- Mark optional keys with ``NotRequired[...]``.
- Streaming SSE event objects each carry the literal ``type`` discriminator
  exactly as it appears on the wire (``response.created`` etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

try:  # Python 3.11+
    from typing import TypedDict, Literal, NotRequired  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover — CI pins Python 3.10
    from typing_extensions import TypedDict, Literal, NotRequired  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Documented deltas from OpenClaw / OpenAI Responses upstream
# ---------------------------------------------------------------------------

RESPONSES_GATEWAY_DELTAS: Dict[str, str] = {
    "tool_result": (
        "Custom output item type; not in OpenAI Responses spec. Carries the "
        "concrete result of an internal Ouroboros tool call so clients can "
        "render execution traces. Standard OpenAI clients treat unknown types "
        "as opaque output items and ignore them."
    ),
    "client_tools_resume": (
        "Client-provided tools (request.tools) are honored via the standard "
        "pause/resume protocol: when the agent calls a client tool, the "
        "response ends with status='incomplete' and the client follows up "
        "with previous_response_id + a function_call_output input item."
    ),
    "internal_tool_visibility": (
        "Internal Ouroboros tools (read_file, write_file, etc.) are surfaced "
        "as standard function_call output items so OpenAI-shape clients can "
        "render which tools the agent invoked. tool_result items follow."
    ),
    "model_namespace": (
        "Accepted model values: 'openclaw', 'openclaw/default', "
        "'openclaw/<agent>'. All of them route to the single Ouroboros agent "
        "in this build. The optional x-openclaw-model header pins the "
        "underlying provider model for the duration of one request."
    ),
}


# ---------------------------------------------------------------------------
# Request — input array items
# ---------------------------------------------------------------------------


class _Source(TypedDict, total=False):
    type: Literal["url", "base64"]
    url: str
    data: str
    media_type: str
    filename: str


class InputMessageContent(TypedDict, total=False):
    type: Literal["input_text", "output_text", "input_image", "input_file"]
    text: str
    image_url: str
    source: _Source


class InputMessageItem(TypedDict, total=False):
    type: Literal["message"]
    role: Literal["system", "developer", "user", "assistant"]
    content: Union[str, List[InputMessageContent]]


class InputImageItem(TypedDict, total=False):
    type: Literal["input_image"]
    source: _Source


class InputFileItem(TypedDict, total=False):
    type: Literal["input_file"]
    source: _Source


class FunctionCallOutputItem(TypedDict, total=False):
    """Carries a tool result back from the client as part of the resume protocol."""
    type: Literal["function_call_output"]
    call_id: str
    output: str


InputItem = Union[
    InputMessageItem,
    InputImageItem,
    InputFileItem,
    FunctionCallOutputItem,
]


# ---------------------------------------------------------------------------
# Request — tool definitions (client tools)
# ---------------------------------------------------------------------------


class FunctionToolDef(TypedDict, total=False):
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolDef(TypedDict, total=False):
    type: Literal["function"]
    function: FunctionToolDef


# ---------------------------------------------------------------------------
# Request body
# ---------------------------------------------------------------------------


class CreateResponseRequest(TypedDict, total=False):
    model: str
    input: Union[str, List[InputItem]]
    instructions: str
    tools: List[ToolDef]
    tool_choice: Union[str, Dict[str, Any]]
    stream: bool
    max_output_tokens: int
    user: str
    previous_response_id: str
    # Accepted but currently ignored (mirrors OpenClaw's documented behavior).
    max_tool_calls: int
    reasoning: Any
    metadata: Dict[str, Any]
    store: Dict[str, Any]
    truncation: Any


# ---------------------------------------------------------------------------
# Response — output items
# ---------------------------------------------------------------------------


class OutputTextContentPart(TypedDict, total=False):
    type: Literal["output_text"]
    text: str
    annotations: List[Any]


class MessageOutputItem(TypedDict, total=False):
    type: Literal["message"]
    id: str
    status: Literal["in_progress", "completed", "incomplete"]
    role: Literal["assistant"]
    content: List[OutputTextContentPart]


class FunctionCallOutputItemEcho(TypedDict, total=False):
    type: Literal["function_call"]
    id: str
    call_id: str
    name: str
    arguments: str
    status: Literal["in_progress", "completed", "incomplete"]


class ToolResultOutputItem(TypedDict, total=False):
    """Custom OpenClaw extension — not in upstream OpenAI spec.

    See ``RESPONSES_GATEWAY_DELTAS['tool_result']``.
    """
    type: Literal["tool_result"]
    id: str
    call_id: str
    output: str
    is_error: NotRequired[bool]


OutputItem = Union[MessageOutputItem, FunctionCallOutputItemEcho, ToolResultOutputItem]


# ---------------------------------------------------------------------------
# Response object
# ---------------------------------------------------------------------------


class Usage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponseObject(TypedDict, total=False):
    id: str
    object: Literal["response"]
    created_at: int
    status: Literal["completed", "failed", "in_progress", "incomplete"]
    model: str
    output: List[OutputItem]
    output_text: str
    usage: Usage
    previous_response_id: NotRequired[str]
    user: NotRequired[str]
    error: NotRequired[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Streaming SSE events — names are exactly what we emit on the wire.
# ---------------------------------------------------------------------------


# Event names (constants — keep in sync with translator.py)
EVENT_CREATED = "response.created"
EVENT_IN_PROGRESS = "response.in_progress"
EVENT_OUTPUT_ITEM_ADDED = "response.output_item.added"
EVENT_CONTENT_PART_ADDED = "response.content_part.added"
EVENT_OUTPUT_TEXT_DELTA = "response.output_text.delta"
EVENT_OUTPUT_TEXT_DONE = "response.output_text.done"
EVENT_CONTENT_PART_DONE = "response.content_part.done"
EVENT_OUTPUT_ITEM_DONE = "response.output_item.done"
EVENT_FUNCTION_ARGS_DELTA = "response.function_call_arguments.delta"
EVENT_FUNCTION_ARGS_DONE = "response.function_call_arguments.done"
EVENT_COMPLETED = "response.completed"
EVENT_FAILED = "response.failed"

ALL_STREAM_EVENTS = (
    EVENT_CREATED,
    EVENT_IN_PROGRESS,
    EVENT_OUTPUT_ITEM_ADDED,
    EVENT_CONTENT_PART_ADDED,
    EVENT_OUTPUT_TEXT_DELTA,
    EVENT_OUTPUT_TEXT_DONE,
    EVENT_CONTENT_PART_DONE,
    EVENT_OUTPUT_ITEM_DONE,
    EVENT_FUNCTION_ARGS_DELTA,
    EVENT_FUNCTION_ARGS_DONE,
    EVENT_COMPLETED,
    EVENT_FAILED,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorBody(TypedDict, total=False):
    message: str
    type: Literal[
        "invalid_request_error",
        "authentication_error",
        "permission_error",
        "not_found_error",
        "rate_limit_error",
        "server_error",
    ]
    code: NotRequired[str]
    param: NotRequired[str]


class ErrorEnvelope(TypedDict):
    error: ErrorBody


# ---------------------------------------------------------------------------
# OpenClaw-specific request headers we honor
# ---------------------------------------------------------------------------

OPENCLAW_HEADERS = (
    "x-openclaw-agent-id",
    "x-openclaw-model",
    "x-openclaw-session-key",
    "x-openclaw-message-channel",
    "x-openclaw-scopes",
)
