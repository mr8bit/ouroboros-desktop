"""Resolve ``input_image`` / ``input_file`` items from /v1/responses requests.

Returns ``ResolvedAttachment`` objects ready to ride alongside (images) or
inside (text/PDF content) the bridge message.  Mirrors OpenClaw's
documented limits and protections:

- Per-modality byte caps (configurable via ``OUROBOROS_RESPONSES_FILES_MAX_BYTES``
  and ``OUROBOROS_RESPONSES_IMAGES_MAX_BYTES``).
- MIME allowlists matching the OpenClaw spec.
- URL fetches refuse private / loopback / link-local IPs and honour an
  optional comma-separated hostname allowlist.
- Redirect cap (default 3), timeout (default 10s).
- PDFs: best-effort text extraction via ``pypdf``.  Optional dependency —
  when missing, PDFs are skipped with a clearly logged warning instead of
  failing the whole request.

This module never invokes the LLM; it just shapes content for the bridge.
All raised exceptions surface as 400 errors via the HTTP handler.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import logging
import os
import socket
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REQUESTS_AVAILABLE = False
    requests = None  # type: ignore[assignment]

from ouroboros.responses_translator import ResolvedAttachment, wrap_untrusted

log = logging.getLogger("responses-server")


# OpenClaw-documented MIME allowlists.
ALLOWED_IMAGE_MIMES = (
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/heic",
    "image/heif",
)

ALLOWED_FILE_MIMES = (
    "text/plain",
    "text/markdown",
    "text/html",
    "text/csv",
    "application/json",
    "application/pdf",
)

# Per-modality defaults (matching OpenClaw documentation).
DEFAULT_FILE_MAX_BYTES = 5_000_000
DEFAULT_IMAGE_MAX_BYTES = 10_000_000
DEFAULT_FILE_MAX_CHARS = 200_000
DEFAULT_PDF_MAX_PAGES = 4
DEFAULT_PDF_MIN_TEXT_CHARS = 200
DEFAULT_TIMEOUT_SEC = 10
DEFAULT_MAX_REDIRECTS = 3


class AttachmentRejected(ValueError):
    """Raised when an attachment fails validation (size, mime, IP, etc.)."""


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _hostname_allowlist() -> Optional[set[str]]:
    raw = (os.environ.get("OUROBOROS_RESPONSES_URL_ALLOWLIST") or "").strip()
    if not raw:
        return None
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


# ---------------------------------------------------------------------------
# Network safety
# ---------------------------------------------------------------------------


def _refuse_private_ips(hostname: str) -> None:
    """Raise AttachmentRejected if hostname resolves to a non-public IP.

    SSRF defence — refuse loopback, link-local, multicast, private, and
    reserved ranges.  Resolved through the OS resolver; we accept the
    resolution at face value.  Even if the host has split-horizon DNS the
    public IP path is still checked (defence in depth, not perfect).
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise AttachmentRejected(f"DNS lookup failed for {hostname}: {exc}") from exc
    for family, _, _, _, sockaddr in infos:
        ip_str = sockaddr[0] if sockaddr else ""
        if not ip_str:
            continue
        try:
            ip = ipaddress.ip_address(ip_str.split("%", 1)[0])  # strip zone id
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise AttachmentRejected(
                f"Refusing to fetch from non-public address {ip} (host={hostname})"
            )


def _check_url_allowlist(url: str) -> str:
    """Return the hostname after allowlist + scheme checks."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise AttachmentRejected(f"Unsupported URL scheme '{parsed.scheme}' (use http/https)")
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise AttachmentRejected("URL is missing a hostname")
    allow = _hostname_allowlist()
    if allow and hostname not in allow:
        # Allow exact match or a domain suffix match (".example.com")
        if not any(hostname == h or hostname.endswith("." + h) for h in allow):
            raise AttachmentRejected(f"Host '{hostname}' is not in the URL allowlist")
    return hostname


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------


def _extract_source(item: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Pull the `source` block out of an input item.

    Accepts both shapes seen in the wild:
        {"source": {"type": "url", "url": "..."}}
        {"image_url": "..."}    (older OpenAI form)
    """
    source = item.get("source")
    if isinstance(source, dict):
        return str(source.get("type") or "").strip(), source
    image_url = item.get("image_url")
    if isinstance(image_url, str) and image_url:
        return "url", {"type": "url", "url": image_url}
    if isinstance(image_url, dict):
        return "url", {"type": "url", "url": str(image_url.get("url") or "")}
    file_url = item.get("file_url")
    if isinstance(file_url, str) and file_url:
        return "url", {"type": "url", "url": file_url}
    return "", {}


# ---------------------------------------------------------------------------
# URL fetch
# ---------------------------------------------------------------------------


def _fetch_url(url: str, *, max_bytes: int) -> Tuple[bytes, str]:
    """Return (body, content_type) after redirect-and-size checks.

    Raises AttachmentRejected on any policy violation.  Content-Type is
    returned lowercased; missing / malformed headers default to
    ``application/octet-stream`` so the caller can gate on the MIME
    allowlist.
    """
    if not _REQUESTS_AVAILABLE:
        raise AttachmentRejected("URL fetching requires the 'requests' package")
    if not _env_bool("OUROBOROS_RESPONSES_URL_FETCH", True):
        raise AttachmentRejected("URL fetching is disabled by configuration")

    timeout = _env_int("OUROBOROS_RESPONSES_URL_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC)
    max_redirects = _env_int("OUROBOROS_RESPONSES_URL_MAX_REDIRECTS", DEFAULT_MAX_REDIRECTS)

    current_url = url
    seen: List[str] = []
    for _ in range(max_redirects + 1):
        hostname = _check_url_allowlist(current_url)
        _refuse_private_ips(hostname)
        seen.append(current_url)
        # Manual redirect handling so each hop re-runs the IP/allowlist check.
        resp = requests.get(  # type: ignore[union-attr]
            current_url,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
            headers={"User-Agent": "ouroboros-responses-gateway/1"},
        )
        if 300 <= resp.status_code < 400:
            location = resp.headers.get("Location") or resp.headers.get("location")
            resp.close()
            if not location:
                raise AttachmentRejected(f"Redirect with no Location header at {current_url}")
            from urllib.parse import urljoin
            current_url = urljoin(current_url, location)
            if current_url in seen:
                raise AttachmentRejected(f"Redirect loop at {current_url}")
            continue
        if resp.status_code >= 400:
            resp.close()
            raise AttachmentRejected(f"HTTP {resp.status_code} fetching {current_url}")
        # Read with byte cap.
        declared_len = resp.headers.get("Content-Length")
        if declared_len and declared_len.isdigit() and int(declared_len) > max_bytes:
            resp.close()
            raise AttachmentRejected(f"Resource exceeds {max_bytes} bytes (Content-Length)")
        chunks: List[bytes] = []
        total = 0
        try:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise AttachmentRejected(f"Resource exceeds {max_bytes} bytes")
        finally:
            resp.close()
        body = b"".join(chunks)
        ctype = (resp.headers.get("Content-Type") or "application/octet-stream").split(";", 1)[0].strip().lower()
        return body, ctype
    raise AttachmentRejected(f"Too many redirects (>{max_redirects}) for {url}")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _decode_base64(data: str, *, max_bytes: int) -> bytes:
    if not isinstance(data, str) or not data:
        raise AttachmentRejected("Empty base64 payload")
    cleaned = data.strip().replace("\n", "").replace("\r", "")
    # Strip optional data: URI prefix.
    if cleaned.startswith("data:"):
        comma = cleaned.find(",")
        if comma == -1:
            raise AttachmentRejected("Malformed data: URI in base64 payload")
        cleaned = cleaned[comma + 1:]
    try:
        decoded = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AttachmentRejected(f"Invalid base64 payload: {exc}") from exc
    if len(decoded) > max_bytes:
        raise AttachmentRejected(f"Decoded payload exceeds {max_bytes} bytes")
    return decoded


def _to_b64(body: bytes) -> str:
    return base64.b64encode(body).decode("ascii")


# ---------------------------------------------------------------------------
# PDF handling
# ---------------------------------------------------------------------------


def _try_extract_pdf_text(body: bytes, *, max_pages: int, max_chars: int) -> str:
    """Best-effort PDF→text via pypdf.  Returns "" if pypdf is missing."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        log.info(
            "pypdf not installed — PDF input_file will be skipped. "
            "Install with `pip install pypdf` to enable text extraction.",
        )
        return ""
    import io
    try:
        reader = PdfReader(io.BytesIO(body))
    except Exception as exc:
        log.warning("Failed to parse PDF: %s", exc)
        return ""
    out: List[str] = []
    used = 0
    for idx, page in enumerate(reader.pages):
        if idx >= max_pages:
            break
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text:
            continue
        out.append(text)
        used += len(text)
        if used >= max_chars:
            break
    joined = "\n".join(out).strip()
    return joined[:max_chars]


# ---------------------------------------------------------------------------
# Public resolvers
# ---------------------------------------------------------------------------


def resolve_input_image(item: Dict[str, Any]) -> Optional[ResolvedAttachment]:
    """Resolve one input_image item into a base64 image attachment."""
    src_type, source = _extract_source(item)
    max_bytes = _env_int("OUROBOROS_RESPONSES_IMAGES_MAX_BYTES", DEFAULT_IMAGE_MAX_BYTES)
    if src_type == "base64":
        media_type = str(source.get("media_type") or "").strip().lower()
        if media_type not in ALLOWED_IMAGE_MIMES:
            raise AttachmentRejected(f"Image media_type '{media_type}' not allowed")
        body = _decode_base64(str(source.get("data") or ""), max_bytes=max_bytes)
        filename = str(source.get("filename") or "")
        return ResolvedAttachment(
            kind="image",
            body=_to_b64(body),
            media_type=media_type,
            filename=filename,
        )
    if src_type == "url":
        url = str(source.get("url") or "").strip()
        if not url:
            raise AttachmentRejected("input_image url source is empty")
        body, ctype = _fetch_url(url, max_bytes=max_bytes)
        if ctype not in ALLOWED_IMAGE_MIMES:
            raise AttachmentRejected(f"Fetched image content-type '{ctype}' not allowed")
        return ResolvedAttachment(
            kind="image",
            body=_to_b64(body),
            media_type=ctype,
            filename=url.rsplit("/", 1)[-1][:120],
        )
    raise AttachmentRejected(f"Unsupported input_image source type '{src_type}'")


def resolve_input_file(item: Dict[str, Any]) -> Optional[ResolvedAttachment]:
    """Resolve one input_file item into either a wrapped-text or image attachment.

    Text/JSON/HTML/CSV/Markdown files become text attachments.  PDFs are
    extracted to text via pypdf when available; if pypdf is missing or the
    PDF has no extractable text the file is skipped with a marker comment.
    """
    src_type, source = _extract_source(item)
    max_bytes = _env_int("OUROBOROS_RESPONSES_FILES_MAX_BYTES", DEFAULT_FILE_MAX_BYTES)
    max_chars = _env_int("OUROBOROS_RESPONSES_FILES_MAX_CHARS", DEFAULT_FILE_MAX_CHARS)
    pdf_max_pages = _env_int("OUROBOROS_RESPONSES_FILES_PDF_MAX_PAGES", DEFAULT_PDF_MAX_PAGES)
    pdf_min_text = _env_int("OUROBOROS_RESPONSES_FILES_PDF_MIN_TEXT_CHARS", DEFAULT_PDF_MIN_TEXT_CHARS)

    if src_type == "base64":
        media_type = str(source.get("media_type") or "").strip().lower()
        if media_type not in ALLOWED_FILE_MIMES:
            raise AttachmentRejected(f"File media_type '{media_type}' not allowed")
        body = _decode_base64(str(source.get("data") or ""), max_bytes=max_bytes)
        filename = str(source.get("filename") or "file")
    elif src_type == "url":
        url = str(source.get("url") or "").strip()
        if not url:
            raise AttachmentRejected("input_file url source is empty")
        body, media_type = _fetch_url(url, max_bytes=max_bytes)
        if media_type not in ALLOWED_FILE_MIMES:
            raise AttachmentRejected(f"Fetched file content-type '{media_type}' not allowed")
        filename = url.rsplit("/", 1)[-1][:120] or "file"
    else:
        raise AttachmentRejected(f"Unsupported input_file source type '{src_type}'")

    if media_type == "application/pdf":
        text = _try_extract_pdf_text(body, max_pages=pdf_max_pages, max_chars=max_chars)
        if not text or len(text) < pdf_min_text:
            log.info(
                "PDF '%s' produced %d chars of text (< %d threshold) — skipping",
                filename, len(text), pdf_min_text,
            )
            wrapped = wrap_untrusted(
                f"[file: {filename}, application/pdf — extraction yielded too little text]",
            )
            return ResolvedAttachment(kind="text", body=wrapped, media_type=media_type, filename=filename)
        wrapped = wrap_untrusted(f"[file: {filename}, application/pdf]\n{text}")
        return ResolvedAttachment(kind="text", body=wrapped, media_type=media_type, filename=filename)

    # Text-shaped formats — decode directly.
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception as exc:
        raise AttachmentRejected(f"Failed to decode {media_type} as UTF-8: {exc}") from exc
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n[truncated at {max_chars} chars]"
    wrapped = wrap_untrusted(f"[file: {filename}, {media_type}]\n{text}")
    return ResolvedAttachment(kind="text", body=wrapped, media_type=media_type, filename=filename)


def build_resolver():
    """Return a resolver callable suitable for translate_input_to_user_message.

    Dispatches by ``item['type']`` and propagates AttachmentRejected as a
    400-class error to the HTTP handler (the executor wraps ValueError).
    """
    def _resolve(item: Dict[str, Any]) -> Optional[ResolvedAttachment]:
        t = str(item.get("type") or "").strip()
        if t == "input_image":
            return resolve_input_image(item)
        if t == "input_file":
            return resolve_input_file(item)
        return None
    return _resolve
