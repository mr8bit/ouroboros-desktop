"""Tests for the OpenResponses HTTP gateway.

Covers (without spinning up the supervisor / a real LLM):
- responses_translator: input array translation, response object building,
  SSE event ordering.
- responses_session_store: round-trip, pause/resume flags, TTL cleanup.
- responses_files: base64 decoding limits, MIME allowlists, SSRF guard.
- responses_executor: chat_id resolution, deterministic hashing,
  concurrency gate.
- responses_server: HTTP-level auth, body parsing, model validation,
  client-tools rejection.

The agent dispatch path is NOT exercised here — it requires a live
supervisor + LLM provider.  Those paths are covered by integration tests
that bring up the whole runtime.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import pathlib
import time
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# responses_translator
# ---------------------------------------------------------------------------


class TestTranslator:
    def test_string_input_becomes_user_text(self):
        from ouroboros.responses_translator import translate_input_to_user_message
        out = translate_input_to_user_message("hi there")
        assert out.user_text == "hi there"
        assert out.system_prefix == ""
        assert out.attachments == []

    def test_message_array_collects_user_and_system(self):
        from ouroboros.responses_translator import translate_input_to_user_message
        body = [
            {"type": "message", "role": "system", "content": "be brief"},
            {"type": "message", "role": "user", "content": "ping"},
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "extra"},
            ]},
        ]
        out = translate_input_to_user_message(body, instructions="treat answers as terse")
        assert "ping" in out.user_text
        assert "extra" in out.user_text
        assert "be brief" in out.system_prefix
        assert "treat answers as terse" in out.system_prefix

    def test_function_call_outputs_collected(self):
        from ouroboros.responses_translator import translate_input_to_user_message
        body = [
            {"type": "function_call_output", "call_id": "call_1", "output": "42"},
            {"type": "message", "role": "user", "content": "thanks"},
        ]
        out = translate_input_to_user_message(body)
        assert out.function_call_outputs == [("call_1", "42")]

    def test_response_object_orders_function_calls_then_message(self):
        from ouroboros.responses_translator import (
            CapturedToolCall,
            build_response_object,
        )
        tc = CapturedToolCall(
            item_id="fc_1", call_id="call_1", name="read_file",
            arguments_json='{"path":"BIBLE.md"}', result_text="contents...",
        )
        obj = build_response_object(
            response_id="resp_x", model="openclaw",
            final_text="here you go", tool_calls=[tc],
        )
        types = [item["type"] for item in obj["output"]]
        # function_call → tool_result → assistant message
        assert types == ["function_call", "tool_result", "message"]
        assert obj["status"] == "completed"
        assert obj["object"] == "response"
        assert obj["output_text"] == "here you go"

    def test_sse_initial_events_emit_created_and_in_progress(self):
        from ouroboros.responses_translator import sse_initial_events
        frames = list(sse_initial_events(response_id="resp_x", model="openclaw"))
        assert any("event: response.created" in f for f in frames)
        assert any("event: response.in_progress" in f for f in frames)

    def test_sse_function_call_full_sequence(self):
        from ouroboros.responses_translator import (
            CapturedToolCall, sse_function_call,
        )
        tc = CapturedToolCall(
            item_id="fc_1", call_id="call_1", name="run_shell",
            arguments_json='{"cmd":"ls"}',
        )
        seq = list(sse_function_call(tc, output_index=0))
        markers = [
            "event: response.output_item.added",
            "event: response.function_call_arguments.delta",
            "event: response.function_call_arguments.done",
            "event: response.output_item.done",
        ]
        for marker in markers:
            assert any(marker in f for f in seq), f"missing {marker}"

    def test_sse_message_text_chunks_into_deltas(self):
        from ouroboros.responses_translator import sse_message_text
        seq = list(sse_message_text("hello world", output_index=0, chunk_size=4))
        deltas = [f for f in seq if "event: response.output_text.delta" in f]
        # "hello world" / 4-char chunks → 3 deltas
        assert len(deltas) == 3

    def test_wrap_untrusted_uses_openclaw_markers(self):
        from ouroboros.responses_translator import wrap_untrusted
        wrapped = wrap_untrusted("secret payload", ext_id="ext_42")
        assert wrapped.startswith('<<<EXTERNAL_UNTRUSTED_CONTENT id="ext_42">>>')
        assert wrapped.endswith('<<<END id="ext_42">>>')

    def test_wrap_untrusted_auto_id(self):
        from ouroboros.responses_translator import wrap_untrusted
        a = wrap_untrusted("x")
        b = wrap_untrusted("x")
        # Two calls should produce distinct ids — guarantees the marker can't
        # be forged by repeating the previous block verbatim.
        assert a != b
        assert "EXTERNAL_UNTRUSTED_CONTENT" in a

    def test_instructions_only_yields_system_prefix(self):
        from ouroboros.responses_translator import translate_input_to_user_message
        out = translate_input_to_user_message([], instructions="be terse")
        assert out.user_text == ""
        assert out.system_prefix == "be terse"

    def test_developer_role_routed_to_system(self):
        from ouroboros.responses_translator import translate_input_to_user_message
        out = translate_input_to_user_message([
            {"type": "message", "role": "developer", "content": "no jokes"},
            {"type": "message", "role": "user", "content": "hi"},
        ])
        assert "no jokes" in out.system_prefix
        assert "[developer]" in out.system_prefix

    def test_function_call_output_with_dict_serialised(self):
        from ouroboros.responses_translator import translate_input_to_user_message
        out = translate_input_to_user_message([
            {"type": "function_call_output", "call_id": "c1", "output": {"answer": 42}},
        ])
        assert out.function_call_outputs == [("c1", json.dumps({"answer": 42}, ensure_ascii=False))]

    def test_unknown_item_types_silently_ignored(self):
        from ouroboros.responses_translator import translate_input_to_user_message
        # OpenClaw spec says reasoning/item_reference are accepted but ignored.
        out = translate_input_to_user_message([
            {"type": "reasoning", "summary": "..."},
            {"type": "item_reference", "id": "msg_123"},
            {"type": "message", "role": "user", "content": "ok"},
        ])
        assert out.user_text == "ok"

    def test_sse_format_blank_line_terminator(self):
        from ouroboros.responses_translator import sse_format
        frame = sse_format("response.created", {"x": 1})
        # Two newlines required at the end (event terminator).
        assert frame.endswith("\n\n")
        assert frame.startswith("event: response.created\n")
        # Data line is JSON parseable.
        data_line = [line for line in frame.split("\n") if line.startswith("data: ")][0]
        assert json.loads(data_line[len("data: "):]) == {"x": 1}

    def test_sse_done_marker(self):
        from ouroboros.responses_translator import sse_done
        assert sse_done() == "data: [DONE]\n\n"

    def test_sse_completed_wraps_response(self):
        from ouroboros.responses_translator import (
            build_response_object, sse_completed,
        )
        obj = build_response_object(
            response_id="resp_x", model="openclaw",
            final_text="ok", tool_calls=[],
        )
        frame = sse_completed(obj)
        assert "event: response.completed" in frame
        body = json.loads(frame.split("data: ", 1)[1].rstrip())
        assert body["response"]["output_text"] == "ok"

    def test_sse_failed_carries_error_body(self):
        from ouroboros.responses_translator import sse_failed
        frame = sse_failed({"message": "boom", "type": "server_error"})
        assert "event: response.failed" in frame
        body = json.loads(frame.split("data: ", 1)[1].rstrip())
        assert body["error"]["message"] == "boom"

    def test_response_object_with_no_text_only_tool_calls(self):
        from ouroboros.responses_translator import (
            CapturedToolCall, build_response_object,
        )
        tc = CapturedToolCall(
            item_id="fc_1", call_id="call_1", name="x",
            arguments_json="{}", result_text="",
        )
        obj = build_response_object(
            response_id="r", model="openclaw",
            final_text="", tool_calls=[tc],
        )
        types = [item["type"] for item in obj["output"]]
        # No tool_result item when result_text is empty, no message item when text is empty.
        assert types == ["function_call"]

    def test_response_object_passes_user_and_previous(self):
        from ouroboros.responses_translator import build_response_object
        obj = build_response_object(
            response_id="r", model="openclaw",
            final_text="hi", tool_calls=[],
            previous_response_id="resp_prev", user_field="alice",
        )
        assert obj["previous_response_id"] == "resp_prev"
        assert obj["user"] == "alice"

    def test_id_generators_have_expected_prefixes(self):
        from ouroboros.responses_translator import (
            new_response_id, new_message_id,
            new_function_call_item_id, new_function_call_call_id,
            new_tool_result_item_id,
        )
        assert new_response_id().startswith("resp_")
        assert new_message_id().startswith("msg_")
        assert new_function_call_item_id().startswith("fc_")
        assert new_function_call_call_id().startswith("call_")
        assert new_tool_result_item_id().startswith("tr_")

    def test_message_text_with_empty_string_skips_deltas(self):
        from ouroboros.responses_translator import sse_message_text
        seq = list(sse_message_text("", output_index=0))
        # No delta events when there's no text — but the message frame must
        # still open and close so the client sees a complete sequence.
        assert not any("output_text.delta" in f for f in seq)
        assert any("output_item.added" in f for f in seq)
        assert any("output_item.done" in f for f in seq)


# ---------------------------------------------------------------------------
# responses_session_store
# ---------------------------------------------------------------------------


class TestSessionStore:
    def _make(self, tmp_path):
        from ouroboros.responses_session_store import FileResponsesSessionStore
        return FileResponsesSessionStore(tmp_path / "data", ttl_hours=24)

    def test_upsert_and_lookup(self, tmp_path):
        store = self._make(tmp_path)
        store.upsert(
            response_id="resp_a",
            virtual_chat_id=-1_500_000_000,
            user_field="alice",
            session_key=None,
        )
        rec = store.lookup("resp_a")
        assert rec is not None
        assert rec["virtual_chat_id"] == -1_500_000_000
        assert rec["user"] == "alice"

    def test_unknown_lookup_returns_none(self, tmp_path):
        store = self._make(tmp_path)
        assert store.lookup("resp_missing") is None

    def test_pause_and_resume_flags(self, tmp_path):
        store = self._make(tmp_path)
        store.upsert(response_id="resp_b", virtual_chat_id=-1)
        store.mark_paused("resp_b", pending_call_id="call_1")
        assert store.lookup("resp_b")["paused_for_tool_call"] is True
        store.mark_resumed("resp_b")
        assert store.lookup("resp_b")["paused_for_tool_call"] is False

    def test_path_traversal_rejected(self, tmp_path):
        store = self._make(tmp_path)
        # Both upsert and lookup should normalise path-traversal attempts so
        # nothing escapes the store directory.  Filename may still contain
        # dots after sanitisation, but it must land inside the store dir.
        store.upsert(response_id="../../etc/passwd", virtual_chat_id=-1)
        store_dir = tmp_path / "data" / "responses_sessions"
        landing = list(store_dir.glob("*.json"))
        assert landing, "expected at least one file written"
        for p in landing:
            resolved = p.resolve()
            # Resolved path must still be inside the store directory.
            assert store_dir.resolve() in resolved.parents

    def test_cleanup_expired_removes_old_files(self, tmp_path):
        from ouroboros.responses_session_store import FileResponsesSessionStore
        store = FileResponsesSessionStore(tmp_path / "data", ttl_hours=1)
        store.upsert(response_id="resp_old", virtual_chat_id=-1)
        # Backdate the file 2 hours.
        path = (tmp_path / "data" / "responses_sessions").glob("*.json")
        for p in path:
            old = time.time() - 7200
            import os
            os.utime(p, (old, old))
        removed = store.cleanup_expired()
        assert removed >= 1

    def test_cleanup_returns_zero_when_fresh(self, tmp_path):
        store = self._make(tmp_path)
        store.upsert(response_id="resp_fresh", virtual_chat_id=-1)
        assert store.cleanup_expired() == 0

    def test_pause_resume_records_pending_call_id(self, tmp_path):
        store = self._make(tmp_path)
        store.upsert(response_id="resp_p", virtual_chat_id=-7)
        store.mark_paused("resp_p", pending_call_id="call_xyz")
        rec = store.lookup("resp_p")
        assert rec["paused_for_tool_call"] is True
        assert rec["pending_call_id"] == "call_xyz"
        store.mark_resumed("resp_p")
        rec = store.lookup("resp_p")
        assert rec["paused_for_tool_call"] is False
        assert rec["pending_call_id"] is None

    def test_upsert_preserves_created_at(self, tmp_path):
        store = self._make(tmp_path)
        store.upsert(response_id="resp_c", virtual_chat_id=-1)
        first_created = store.lookup("resp_c")["created_at"]
        time.sleep(0.01)
        store.upsert(response_id="resp_c", virtual_chat_id=-1)
        rec = store.lookup("resp_c")
        # created_at must not change on subsequent upserts.
        assert rec["created_at"] == first_created
        # last_used_at must move forward.
        assert rec["last_used_at"] >= first_created

    def test_delete_removes_entry(self, tmp_path):
        store = self._make(tmp_path)
        store.upsert(response_id="resp_d", virtual_chat_id=-1)
        assert store.lookup("resp_d") is not None
        store.delete("resp_d")
        assert store.lookup("resp_d") is None

    def test_corrupted_file_returns_none(self, tmp_path):
        store = self._make(tmp_path)
        store.upsert(response_id="resp_x", virtual_chat_id=-1)
        # Corrupt the file on disk.
        path = next((tmp_path / "data" / "responses_sessions").glob("*.json"))
        path.write_text("{not valid json", encoding="utf-8")
        # lookup must not raise — corrupted records read as missing.
        assert store.lookup("resp_x") is None


# ---------------------------------------------------------------------------
# responses_files
# ---------------------------------------------------------------------------


class TestFilesGuards:
    def test_base64_image_round_trip(self):
        from ouroboros.responses_files import resolve_input_image
        png_1px = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgYAAAAAMAAS"
            "fXTzMAAAAASUVORK5CYII="
        )
        item = {
            "type": "input_image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(png_1px).decode("ascii"),
                "filename": "px.png",
            },
        }
        out = resolve_input_image(item)
        assert out is not None
        assert out.kind == "image"
        assert out.media_type == "image/png"

    def test_image_mime_rejected(self):
        from ouroboros.responses_files import resolve_input_image, AttachmentRejected
        item = {
            "type": "input_image",
            "source": {"type": "base64", "media_type": "application/x-msdownload", "data": "AA=="},
        }
        with pytest.raises(AttachmentRejected):
            resolve_input_image(item)

    def test_size_limit_enforced(self, monkeypatch):
        from ouroboros.responses_files import resolve_input_image, AttachmentRejected
        monkeypatch.setenv("OUROBOROS_RESPONSES_IMAGES_MAX_BYTES", "10")
        big = base64.b64encode(b"a" * 1000).decode("ascii")
        item = {
            "type": "input_image",
            "source": {"type": "base64", "media_type": "image/png", "data": big},
        }
        with pytest.raises(AttachmentRejected):
            resolve_input_image(item)

    def test_text_file_wrapped_with_untrusted_markers(self):
        from ouroboros.responses_files import resolve_input_file
        item = {
            "type": "input_file",
            "source": {
                "type": "base64",
                "media_type": "text/plain",
                "data": base64.b64encode(b"hello").decode("ascii"),
                "filename": "hi.txt",
            },
        }
        out = resolve_input_file(item)
        assert out is not None
        assert out.kind == "text"
        assert "EXTERNAL_UNTRUSTED_CONTENT" in out.body
        assert "hello" in out.body

    def test_loopback_url_rejected(self):
        from ouroboros.responses_files import _refuse_private_ips, AttachmentRejected
        with pytest.raises(AttachmentRejected):
            _refuse_private_ips("localhost")

    def test_url_scheme_must_be_http(self):
        from ouroboros.responses_files import _check_url_allowlist, AttachmentRejected
        with pytest.raises(AttachmentRejected):
            _check_url_allowlist("file:///etc/passwd")
        with pytest.raises(AttachmentRejected):
            _check_url_allowlist("ftp://example.com/x")

    def test_data_uri_prefix_in_base64_handled(self):
        from ouroboros.responses_files import resolve_input_image
        # Standard 1-px PNG with full data: URI prefix.
        png_b64 = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgYAAAAAMA"
            "ASfXTzMAAAAASUVORK5CYII="
        )
        item = {
            "type": "input_image",
            "source": {"type": "base64", "media_type": "image/png", "data": png_b64},
        }
        out = resolve_input_image(item)
        assert out is not None
        assert out.kind == "image"
        assert out.media_type == "image/png"

    def test_invalid_base64_rejected(self):
        from ouroboros.responses_files import resolve_input_image, AttachmentRejected
        item = {
            "type": "input_image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "not-valid-base64-!!!",
            },
        }
        with pytest.raises(AttachmentRejected):
            resolve_input_image(item)

    def test_unsupported_source_type_rejected(self):
        from ouroboros.responses_files import resolve_input_image, AttachmentRejected
        item = {"type": "input_image", "source": {"type": "ipfs", "cid": "bafy..."}}
        with pytest.raises(AttachmentRejected):
            resolve_input_image(item)

    def test_text_file_truncated_at_max_chars(self, monkeypatch):
        from ouroboros.responses_files import resolve_input_file
        monkeypatch.setenv("OUROBOROS_RESPONSES_FILES_MAX_CHARS", "16")
        big = base64.b64encode(b"abcdefghijklmnopqrstuvwxyz").decode("ascii")
        item = {
            "type": "input_file",
            "source": {
                "type": "base64", "media_type": "text/plain",
                "data": big, "filename": "long.txt",
            },
        }
        out = resolve_input_file(item)
        assert "[truncated at 16 chars]" in out.body

    def test_unsupported_file_mime_rejected(self):
        from ouroboros.responses_files import resolve_input_file, AttachmentRejected
        item = {
            "type": "input_file",
            "source": {
                "type": "base64", "media_type": "application/x-tar",
                "data": base64.b64encode(b"x").decode("ascii"),
            },
        }
        with pytest.raises(AttachmentRejected):
            resolve_input_file(item)

    def test_file_csv_decoded_as_text(self):
        from ouroboros.responses_files import resolve_input_file
        item = {
            "type": "input_file",
            "source": {
                "type": "base64", "media_type": "text/csv",
                "data": base64.b64encode(b"a,b\n1,2\n").decode("ascii"),
                "filename": "data.csv",
            },
        }
        out = resolve_input_file(item)
        assert out.kind == "text"
        assert "a,b" in out.body
        assert "data.csv" in out.body

    def test_url_allowlist_accepts_subdomain(self, monkeypatch):
        from ouroboros.responses_files import _check_url_allowlist
        monkeypatch.setenv("OUROBOROS_RESPONSES_URL_ALLOWLIST", "example.com,images.test")
        # Direct match
        assert _check_url_allowlist("https://example.com/x") == "example.com"
        # Subdomain match through allowlist suffix logic
        assert _check_url_allowlist("https://api.example.com/y") == "api.example.com"

    def test_url_allowlist_rejects_other_host(self, monkeypatch):
        from ouroboros.responses_files import _check_url_allowlist, AttachmentRejected
        monkeypatch.setenv("OUROBOROS_RESPONSES_URL_ALLOWLIST", "example.com")
        with pytest.raises(AttachmentRejected):
            _check_url_allowlist("https://evil.test/x")

    def test_pdf_without_pypdf_returns_skip_marker(self, monkeypatch):
        from ouroboros.responses_files import resolve_input_file
        # Force the pypdf import inside _try_extract_pdf_text to fail.
        import sys
        monkeypatch.setitem(sys.modules, "pypdf", None)
        item = {
            "type": "input_file",
            "source": {
                "type": "base64", "media_type": "application/pdf",
                "data": base64.b64encode(b"%PDF-1.4 stub").decode("ascii"),
                "filename": "a.pdf",
            },
        }
        out = resolve_input_file(item)
        assert out.kind == "text"
        assert "extraction yielded too little text" in out.body or "application/pdf" in out.body

    def test_url_fetch_size_limit_via_content_length(self, monkeypatch):
        from ouroboros.responses_files import _fetch_url, AttachmentRejected

        class _FakeResp:
            status_code = 200
            headers = {"Content-Length": "9999999", "Content-Type": "image/png"}
            def iter_content(self, chunk_size):
                yield b""
            def close(self):
                pass

        # Bypass DNS / IP guards.
        monkeypatch.setattr(
            "ouroboros.responses_files._refuse_private_ips", lambda host: None,
        )
        monkeypatch.setattr(
            "ouroboros.responses_files._check_url_allowlist", lambda url: "example.com",
        )
        monkeypatch.setattr(
            "ouroboros.responses_files.requests.get", lambda *a, **kw: _FakeResp(),
        )
        with pytest.raises(AttachmentRejected) as exc:
            _fetch_url("https://example.com/big.png", max_bytes=1024)
        assert "Content-Length" in str(exc.value)

    def test_url_fetch_redirect_followed_within_limit(self, monkeypatch):
        from ouroboros.responses_files import _fetch_url

        calls = []

        class _Redirect:
            def __init__(self, target):
                self.status_code = 302
                self.headers = {"Location": target, "Content-Type": "text/plain"}
            def iter_content(self, chunk_size):
                yield b""
            def close(self):
                pass

        class _Final:
            status_code = 200
            headers = {"Content-Type": "text/plain"}
            def iter_content(self, chunk_size):
                yield b"final body"
            def close(self):
                pass

        def _fake_get(url, *_, **__):
            calls.append(url)
            if url.endswith("/start"):
                return _Redirect("/middle")
            if url.endswith("/middle"):
                return _Redirect("/end")
            return _Final()

        monkeypatch.setattr("ouroboros.responses_files._refuse_private_ips", lambda host: None)
        monkeypatch.setattr("ouroboros.responses_files._check_url_allowlist", lambda url: "example.com")
        monkeypatch.setattr("ouroboros.responses_files.requests.get", _fake_get)

        body, ctype = _fetch_url("https://example.com/start", max_bytes=10_000)
        assert body == b"final body"
        assert ctype == "text/plain"
        assert len(calls) == 3

    def test_url_fetch_redirect_loop_rejected(self, monkeypatch):
        from ouroboros.responses_files import _fetch_url, AttachmentRejected

        class _Redirect:
            def __init__(self, target):
                self.status_code = 302
                self.headers = {"Location": target, "Content-Type": "text/plain"}
            def iter_content(self, chunk_size):
                yield b""
            def close(self):
                pass

        def _fake_get(url, *_, **__):
            return _Redirect(url)  # always redirects to itself

        monkeypatch.setattr("ouroboros.responses_files._refuse_private_ips", lambda host: None)
        monkeypatch.setattr("ouroboros.responses_files._check_url_allowlist", lambda url: "example.com")
        monkeypatch.setattr("ouroboros.responses_files.requests.get", _fake_get)

        with pytest.raises(AttachmentRejected) as exc:
            _fetch_url("https://example.com/loop", max_bytes=1024)
        assert "loop" in str(exc.value).lower()

    def test_url_fetch_4xx_propagates(self, monkeypatch):
        from ouroboros.responses_files import _fetch_url, AttachmentRejected

        class _NotFound:
            status_code = 404
            headers = {}
            def iter_content(self, chunk_size):
                yield b""
            def close(self):
                pass

        monkeypatch.setattr("ouroboros.responses_files._refuse_private_ips", lambda host: None)
        monkeypatch.setattr("ouroboros.responses_files._check_url_allowlist", lambda url: "example.com")
        monkeypatch.setattr("ouroboros.responses_files.requests.get", lambda *a, **k: _NotFound())

        with pytest.raises(AttachmentRejected) as exc:
            _fetch_url("https://example.com/missing", max_bytes=1024)
        assert "404" in str(exc.value)

    def test_url_fetch_disabled_by_env(self, monkeypatch):
        from ouroboros.responses_files import _fetch_url, AttachmentRejected
        monkeypatch.setenv("OUROBOROS_RESPONSES_URL_FETCH", "false")
        with pytest.raises(AttachmentRejected):
            _fetch_url("https://example.com/x", max_bytes=1024)


# ---------------------------------------------------------------------------
# responses_executor
# ---------------------------------------------------------------------------


class TestExecutorHelpers:
    def test_stable_chat_id_is_deterministic(self):
        from ouroboros.responses_executor import stable_responses_chat_id
        a = stable_responses_chat_id("user:alice")
        b = stable_responses_chat_id("user:alice")
        c = stable_responses_chat_id("user:bob")
        assert a == b
        assert a != c
        # All in the responses negative range.
        assert a <= -1_000_000_000

    def test_resolve_chat_id_prefers_previous_response_id(self, tmp_path):
        from ouroboros.responses_executor import resolve_chat_id
        from ouroboros.responses_session_store import FileResponsesSessionStore
        store = FileResponsesSessionStore(tmp_path / "data", ttl_hours=1)
        store.upsert(response_id="resp_prev", virtual_chat_id=-1_234_567_890)
        chat_id, resumed = resolve_chat_id(
            previous_response_id="resp_prev",
            session_key_header="ignored",
            user_field="ignored",
            session_store=store,
        )
        assert chat_id == -1_234_567_890
        assert resumed is True

    def test_resolve_chat_id_falls_back_to_user_field(self, tmp_path):
        from ouroboros.responses_executor import resolve_chat_id
        chat_id, resumed = resolve_chat_id(
            previous_response_id=None,
            session_key_header=None,
            user_field="bob",
            session_store=None,
        )
        assert resumed is False
        assert chat_id <= -1_000_000_000

    def test_concurrency_gate(self):
        from ouroboros.responses_executor import (
            acquire_slot, configure_concurrency, release_slot,
        )
        configure_concurrency(2)
        assert acquire_slot() is True
        assert acquire_slot() is True
        assert acquire_slot() is False
        release_slot()
        assert acquire_slot() is True
        release_slot()
        release_slot()
        release_slot()

    def test_concurrency_gate_replaceable(self):
        from ouroboros.responses_executor import (
            acquire_slot, configure_concurrency, release_slot,
        )
        configure_concurrency(1)
        assert acquire_slot() is True
        assert acquire_slot() is False
        # Replacing the gate gives a fresh budget — but in-flight slots from
        # the previous gate are forgotten on the caller side.
        configure_concurrency(3)
        assert acquire_slot() is True
        assert acquire_slot() is True
        assert acquire_slot() is True
        assert acquire_slot() is False
        for _ in range(3):
            release_slot()

    def test_resolve_chat_id_session_key_overrides_user(self, tmp_path):
        from ouroboros.responses_executor import (
            resolve_chat_id, stable_responses_chat_id,
        )
        chat_id, resumed = resolve_chat_id(
            previous_response_id=None,
            session_key_header="my-key",
            user_field="alice",
            session_store=None,
        )
        # session-key wins over user_field per OpenClaw spec.
        assert chat_id == stable_responses_chat_id("key:my-key")
        assert resumed is False

    def test_resolve_chat_id_unknown_previous_falls_through(self, tmp_path):
        from ouroboros.responses_executor import resolve_chat_id
        from ouroboros.responses_session_store import FileResponsesSessionStore
        store = FileResponsesSessionStore(tmp_path / "data", ttl_hours=1)
        # No record for the given response_id.
        chat_id, resumed = resolve_chat_id(
            previous_response_id="resp_missing",
            session_key_header=None,
            user_field="bob",
            session_store=store,
        )
        # Falls through to user_field hashing — does not crash.
        assert resumed is False
        assert chat_id <= -1_000_000_000

    def test_next_chat_id_monotonically_decreases(self):
        from ouroboros.responses_executor import next_responses_chat_id
        a = next_responses_chat_id()
        b = next_responses_chat_id()
        c = next_responses_chat_id()
        assert a > b > c
        # All within the responses range.
        assert a <= -1_000_000_000

    def test_next_chat_id_does_not_collide_with_a2a_range(self):
        from ouroboros.responses_executor import next_responses_chat_id
        # A2A starts at -1000 and decrements; many calls must not approach it.
        ids = [next_responses_chat_id() for _ in range(10)]
        assert all(i < -1_000_000_000 for i in ids)

    def test_compose_user_text_combines_system_and_user(self):
        from ouroboros.responses_executor import _compose_user_text
        from ouroboros.responses_translator import TranslatedInput
        translated = TranslatedInput(
            user_text="hello",
            system_prefix="be terse",
        )
        out = _compose_user_text(translated)
        assert "be terse" in out
        assert "hello" in out

    def test_compose_user_text_includes_function_outputs(self):
        from ouroboros.responses_executor import _compose_user_text
        from ouroboros.responses_translator import TranslatedInput
        translated = TranslatedInput(
            user_text="next step",
            system_prefix="",
            function_call_outputs=[("call_1", "42")],
        )
        out = _compose_user_text(translated)
        assert "function_call_output" in out
        assert "call_1" in out
        assert "42" in out
        assert "next step" in out

    def test_coerce_image_data_picks_first_image(self):
        from ouroboros.responses_executor import _coerce_image_data
        from ouroboros.responses_translator import TranslatedInput, ResolvedAttachment
        translated = TranslatedInput(
            user_text="see these",
            system_prefix="",
            attachments=[
                ResolvedAttachment(kind="image", body="AAAA", media_type="image/png", filename="a.png"),
                ResolvedAttachment(kind="image", body="BBBB", media_type="image/jpeg", filename="b.jpg"),
            ],
        )
        out = _coerce_image_data(translated)
        assert out is not None
        assert out[0] == "AAAA"
        assert out[1] == "image/png"
        assert out[2] == "a.png"

    def test_coerce_image_data_returns_none_when_no_images(self):
        from ouroboros.responses_executor import _coerce_image_data
        from ouroboros.responses_translator import TranslatedInput, ResolvedAttachment
        translated = TranslatedInput(
            user_text="x", system_prefix="",
            attachments=[ResolvedAttachment(kind="text", body="...")],
        )
        assert _coerce_image_data(translated) is None

    def test_tool_capture_records_call_and_result(self):
        from ouroboros.responses_executor import ToolCallCapture
        cap = ToolCallCapture(chat_id=-1)
        cap.on_tool_started("call_1", "read_file", {"path": "BIBLE.md"})
        cap.on_tool_finished("call_1", "contents...", is_error=False)
        assert len(cap.calls) == 1
        c = cap.calls[0]
        assert c.name == "read_file"
        assert c.call_id == "call_1"
        assert "BIBLE.md" in c.arguments_json
        assert c.result_text == "contents..."
        assert c.is_error is False

    def test_tool_capture_finished_without_started_is_silent(self):
        from ouroboros.responses_executor import ToolCallCapture
        cap = ToolCallCapture(chat_id=-1)
        # Should not raise — finished events without a matching start are
        # ignored (the started event may arrive late or have been dropped).
        cap.on_tool_finished("unknown_call", "x", is_error=True)
        assert cap.calls == []


# ---------------------------------------------------------------------------
# responses_server (HTTP-level)
# ---------------------------------------------------------------------------


class TestServerHTTP:
    """Exercise the route handlers via Starlette's test client.

    The bridge / agent path is not invoked — we stub it by avoiding requests
    that would dispatch.  Auth + validation paths are full-fidelity.
    """

    def _client(self, monkeypatch, token: str = "secret"):
        starlette = pytest.importorskip("starlette.testclient", reason="starlette not installed")
        from ouroboros.responses_server import _build_app
        if token:
            monkeypatch.setenv("OUROBOROS_RESPONSES_TOKEN", token)
        else:
            monkeypatch.delenv("OUROBOROS_RESPONSES_TOKEN", raising=False)
        app = _build_app()
        return starlette.TestClient(app)

    def test_healthz(self, monkeypatch):
        client = self._client(monkeypatch)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_method_not_allowed(self, monkeypatch):
        client = self._client(monkeypatch)
        resp = client.get("/v1/responses")
        assert resp.status_code == 405
        assert resp.json()["error"]["type"] == "invalid_request_error"

    def test_missing_token_returns_401(self, monkeypatch):
        client = self._client(monkeypatch, token="secret")
        resp = client.post("/v1/responses", json={"model": "openclaw", "input": "hi"})
        assert resp.status_code == 401

    def test_unconfigured_token_returns_503(self, monkeypatch):
        client = self._client(monkeypatch, token="")
        resp = client.post(
            "/v1/responses",
            json={"model": "openclaw", "input": "hi"},
            headers={"Authorization": "Bearer anything"},
        )
        assert resp.status_code == 503

    def test_unknown_model_rejected(self, monkeypatch):
        client = self._client(monkeypatch)
        resp = client.post(
            "/v1/responses",
            json={"model": "gpt-4", "input": "hi"},
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["param"] == "model"

    def test_input_required(self, monkeypatch):
        client = self._client(monkeypatch)
        resp = client.post(
            "/v1/responses",
            json={"model": "openclaw"},
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["param"] == "input"

    def test_client_tools_rejected_in_v1(self, monkeypatch):
        client = self._client(monkeypatch)
        resp = client.post(
            "/v1/responses",
            json={
                "model": "openclaw",
                "input": "hi",
                "tools": [{"type": "function", "function": {"name": "x"}}],
            },
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["param"] == "tools"

    def test_body_too_large(self, monkeypatch):
        monkeypatch.setenv("OUROBOROS_RESPONSES_MAX_BODY_BYTES", "100")
        client = self._client(monkeypatch)
        big_input = "x" * 1000
        resp = client.post(
            "/v1/responses",
            json={"model": "openclaw", "input": big_input},
            headers={
                "Authorization": "Bearer secret",
                "Content-Length": "2000",
            },
        )
        assert resp.status_code == 413

    def test_invalid_json_returns_400(self, monkeypatch):
        client = self._client(monkeypatch)
        resp = client.post(
            "/v1/responses",
            content=b"{not-json",
            headers={
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 400

    def test_empty_body_returns_400(self, monkeypatch):
        client = self._client(monkeypatch)
        resp = client.post(
            "/v1/responses",
            content=b"",
            headers={
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 400

    def test_array_body_rejected(self, monkeypatch):
        client = self._client(monkeypatch)
        resp = client.post(
            "/v1/responses",
            json=["this", "is", "an", "array"],
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 400
        assert "JSON object" in resp.json()["error"]["message"]

    def test_wrong_token_rejected(self, monkeypatch):
        client = self._client(monkeypatch, token="real-secret")
        resp = client.post(
            "/v1/responses",
            json={"model": "openclaw", "input": "hi"},
            headers={"Authorization": "Bearer wrong-secret"},
        )
        assert resp.status_code == 401
        # Standard challenge header per RFC 6750.
        assert resp.headers.get("WWW-Authenticate", "").startswith("Bearer")

    def test_input_must_be_string_or_array(self, monkeypatch):
        client = self._client(monkeypatch)
        resp = client.post(
            "/v1/responses",
            json={"model": "openclaw", "input": 42},
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["param"] == "input"

    def test_openclaw_subagent_model_accepted(self, monkeypatch):
        """`model: openclaw/<id>` must reach validation past model gate.

        The bridge dispatch will fail (no supervisor) — we just want to
        confirm the model namespace is honoured.
        """
        client = self._client(monkeypatch)
        resp = client.post(
            "/v1/responses",
            json={"model": "openclaw/main", "input": "hi"},
            headers={"Authorization": "Bearer secret"},
        )
        # 500 (bridge unavailable) is the expected post-validation failure
        # in this no-supervisor environment — the model wasn't rejected.
        assert resp.status_code in (500, 504)
        assert resp.json()["error"]["type"] == "server_error"

    def test_streaming_emits_failed_when_bridge_unavailable(self, monkeypatch):
        """With no supervisor running, streaming must surface a clean failed
        event followed by [DONE] rather than an HTTP 500.  Clients that
        stream depend on the SSE channel staying open until terminator.
        """
        client = self._client(monkeypatch)
        with client.stream(
            "POST", "/v1/responses",
            json={"model": "openclaw", "input": "hi", "stream": True},
            headers={"Authorization": "Bearer secret"},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = "".join(resp.iter_text())
        # Must include the canonical opening events plus a failed terminator
        # because the supervisor bridge is not running in tests.
        assert "event: response.created" in body
        assert "event: response.in_progress" in body
        assert "event: response.failed" in body
        assert body.rstrip().endswith("data: [DONE]")

    def test_streaming_response_headers(self, monkeypatch):
        """SSE response must disable proxy buffering (X-Accel-Buffering: no)."""
        client = self._client(monkeypatch)
        with client.stream(
            "POST", "/v1/responses",
            json={"model": "openclaw", "input": "hi", "stream": True},
            headers={"Authorization": "Bearer secret"},
        ) as resp:
            assert resp.headers.get("x-accel-buffering") == "no"
            assert resp.headers.get("cache-control") == "no-cache"
            # Drain so the connection closes cleanly.
            for _ in resp.iter_text():
                pass


# ---------------------------------------------------------------------------
# Bridge per-chat-id event subscription (phase 4 plumbing)
# ---------------------------------------------------------------------------


class TestBridgeChatEventSubscription:
    """Verify the new subscribe_chat_events / set_active_chat plumbing.

    Constructs a bare LocalChatBridge and exercises the new methods directly.
    No supervisor / agent involved.
    """

    def test_log_event_routed_to_active_chat_subscriber(self):
        from supervisor.message_bus import LocalChatBridge
        bridge = LocalChatBridge()
        seen: list[dict] = []
        bridge.set_active_chat(-555)
        sub = bridge.subscribe_chat_events(-555, lambda ev: seen.append(ev))
        try:
            bridge.push_log({"type": "tool_call_started", "tool": "read_file", "task_id": "t1"})
            assert len(seen) == 1
            assert seen[0]["type"] == "log"
            assert seen[0]["data"]["type"] == "tool_call_started"
        finally:
            bridge.unsubscribe_chat_events(sub)
            bridge.clear_active_chat()

    def test_subscriber_only_receives_matching_chat_id(self):
        from supervisor.message_bus import LocalChatBridge
        bridge = LocalChatBridge()
        a: list[dict] = []
        b: list[dict] = []
        bridge.subscribe_chat_events(-1, lambda ev: a.append(ev))
        bridge.subscribe_chat_events(-2, lambda ev: b.append(ev))
        bridge.set_active_chat(-1)
        bridge.push_log({"type": "tool_call_started"})
        bridge.set_active_chat(-2)
        bridge.push_log({"type": "tool_call_finished"})
        assert len(a) == 1 and a[0]["data"]["type"] == "tool_call_started"
        assert len(b) == 1 and b[0]["data"]["type"] == "tool_call_finished"

    def test_send_message_fans_out_chat_event(self):
        from supervisor.message_bus import LocalChatBridge
        bridge = LocalChatBridge()
        seen: list[dict] = []
        sub = bridge.subscribe_chat_events(-42, lambda ev: seen.append(ev))
        try:
            bridge.send_message(-42, "hello world")
            assert len(seen) == 1
            assert seen[0]["type"] == "chat"
            assert seen[0]["role"] == "assistant"
            assert seen[0]["content"] == "hello world"
            assert seen[0]["chat_id"] == -42
        finally:
            bridge.unsubscribe_chat_events(sub)

    def test_progress_messages_still_fanout(self):
        """is_progress messages should still reach chat-event subscribers
        (they are filtered only for the A2A response_subs, not for
        chat_event_subs).
        """
        from supervisor.message_bus import LocalChatBridge
        bridge = LocalChatBridge()
        seen: list[dict] = []
        sub = bridge.subscribe_chat_events(-1, lambda ev: seen.append(ev))
        try:
            bridge.send_message(-1, "thinking...", is_progress=True)
            assert len(seen) == 1
            assert seen[0]["is_progress"] is True
        finally:
            bridge.unsubscribe_chat_events(sub)

    def test_typing_action_fanned_out(self):
        from supervisor.message_bus import LocalChatBridge
        bridge = LocalChatBridge()
        seen: list[dict] = []
        sub = bridge.subscribe_chat_events(-9, lambda ev: seen.append(ev))
        try:
            bridge.send_chat_action(-9, "typing")
            assert len(seen) == 1
            assert seen[0]["type"] == "typing"
            assert seen[0]["action"] == "typing"
        finally:
            bridge.unsubscribe_chat_events(sub)

    def test_unsubscribe_stops_delivery(self):
        from supervisor.message_bus import LocalChatBridge
        bridge = LocalChatBridge()
        seen: list[dict] = []
        sub = bridge.subscribe_chat_events(-1, lambda ev: seen.append(ev))
        bridge.send_message(-1, "first")
        bridge.unsubscribe_chat_events(sub)
        bridge.send_message(-1, "second")
        assert len(seen) == 1
        assert seen[0]["content"] == "first"

    def test_clear_active_chat_disables_log_routing(self):
        from supervisor.message_bus import LocalChatBridge
        bridge = LocalChatBridge()
        seen: list[dict] = []
        sub = bridge.subscribe_chat_events(-1, lambda ev: seen.append(ev))
        try:
            bridge.set_active_chat(-1)
            bridge.push_log({"type": "tool_call_started"})
            assert len(seen) == 1
            bridge.clear_active_chat()
            bridge.push_log({"type": "tool_call_finished"})
            # Once cleared, log events without chat_id no longer route.
            assert len(seen) == 1
        finally:
            bridge.unsubscribe_chat_events(sub)

    def test_log_event_with_explicit_chat_id_routes_without_active(self):
        """If the event itself carries chat_id, it routes regardless of the
        bridge's active_chat state."""
        from supervisor.message_bus import LocalChatBridge
        bridge = LocalChatBridge()
        seen: list[dict] = []
        sub = bridge.subscribe_chat_events(-77, lambda ev: seen.append(ev))
        try:
            bridge.clear_active_chat()
            bridge.push_log({"type": "tool_call_started", "chat_id": -77, "tool": "x"})
            assert len(seen) == 1
            assert seen[0]["data"]["chat_id"] == -77
        finally:
            bridge.unsubscribe_chat_events(sub)

    def test_subscriber_callback_exception_does_not_break_others(self):
        from supervisor.message_bus import LocalChatBridge
        bridge = LocalChatBridge()

        def boom(_ev):
            raise RuntimeError("subscriber bug")

        good: list[dict] = []
        bridge.subscribe_chat_events(-1, boom)
        bridge.subscribe_chat_events(-1, lambda ev: good.append(ev))
        bridge.send_message(-1, "still delivered")
        assert len(good) == 1


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------


class TestSchemaConstants:
    def test_event_constants_match_translator_emissions(self):
        """Constants in the schema module must equal the literal strings
        emitted by the translator's SSE helpers — protects against typos
        drifting one side from the other.
        """
        from ouroboros.contracts.openresponses_schema import (
            ALL_STREAM_EVENTS,
            EVENT_CREATED, EVENT_IN_PROGRESS, EVENT_OUTPUT_ITEM_ADDED,
            EVENT_CONTENT_PART_ADDED, EVENT_OUTPUT_TEXT_DELTA,
            EVENT_OUTPUT_TEXT_DONE, EVENT_CONTENT_PART_DONE,
            EVENT_OUTPUT_ITEM_DONE, EVENT_FUNCTION_ARGS_DELTA,
            EVENT_FUNCTION_ARGS_DONE, EVENT_COMPLETED, EVENT_FAILED,
        )
        wire = (
            EVENT_CREATED, EVENT_IN_PROGRESS, EVENT_OUTPUT_ITEM_ADDED,
            EVENT_CONTENT_PART_ADDED, EVENT_OUTPUT_TEXT_DELTA,
            EVENT_OUTPUT_TEXT_DONE, EVENT_CONTENT_PART_DONE,
            EVENT_OUTPUT_ITEM_DONE, EVENT_FUNCTION_ARGS_DELTA,
            EVENT_FUNCTION_ARGS_DONE, EVENT_COMPLETED, EVENT_FAILED,
        )
        # All names start with "response."
        for name in wire:
            assert name.startswith("response."), name
        # All listed events appear in ALL_STREAM_EVENTS.
        assert set(wire) == set(ALL_STREAM_EVENTS)

    def test_openclaw_headers_listed(self):
        from ouroboros.contracts.openresponses_schema import OPENCLAW_HEADERS
        # All expected gateway headers documented per OpenClaw spec.
        for header in (
            "x-openclaw-agent-id", "x-openclaw-model",
            "x-openclaw-session-key", "x-openclaw-message-channel",
            "x-openclaw-scopes",
        ):
            assert header in OPENCLAW_HEADERS

    def test_documented_deltas_present(self):
        from ouroboros.contracts.openresponses_schema import RESPONSES_GATEWAY_DELTAS
        assert "tool_result" in RESPONSES_GATEWAY_DELTAS
        assert "client_tools_resume" in RESPONSES_GATEWAY_DELTAS
        assert "internal_tool_visibility" in RESPONSES_GATEWAY_DELTAS
        assert "model_namespace" in RESPONSES_GATEWAY_DELTAS


# ---------------------------------------------------------------------------
# Streaming path through the executor (no real supervisor)
# ---------------------------------------------------------------------------


class TestStreamingThroughExecutor:
    """Drive ``execute_streaming`` directly with a stubbed bridge call so we
    can validate the full SSE event sequence including tool-call surfacing
    without standing up the supervisor / agent.
    """

    def _consume(self, agen):
        """Drain an async generator into a list of strings."""
        async def _drain():
            return [frame async for frame in agen]
        return asyncio.get_event_loop().run_until_complete(_drain())

    def test_streaming_full_text_only_sequence(self, monkeypatch):
        from ouroboros.responses_executor import execute_streaming, ToolCallCapture
        from ouroboros.responses_translator import TranslatedInput

        async def _stub_bridge(*, chat_id, translated, timeout_sec, capture):
            return "hello there"

        monkeypatch.setattr(
            "ouroboros.responses_executor._run_through_bridge", _stub_bridge,
        )
        gen = execute_streaming(
            response_id="resp_1",
            model="openclaw",
            chat_id=-1_000_000_001,
            translated=TranslatedInput(user_text="hi", system_prefix=""),
            capture=ToolCallCapture(chat_id=-1_000_000_001),
        )
        frames = self._consume(gen)
        # Order check.
        joined = "".join(frames)
        for marker, follower in [
            ("response.created", "response.in_progress"),
            ("response.in_progress", "response.output_item.added"),
            ("response.output_item.added", "response.content_part.added"),
            ("response.content_part.added", "response.output_text.delta"),
            ("response.output_text.delta", "response.output_text.done"),
            ("response.output_text.done", "response.content_part.done"),
            ("response.content_part.done", "response.output_item.done"),
            ("response.output_item.done", "response.completed"),
        ]:
            assert joined.index(marker) < joined.index(follower), \
                f"{marker} should come before {follower}"
        assert joined.rstrip().endswith("data: [DONE]")

    def test_streaming_emits_function_call_events_when_capture_has_call(self, monkeypatch):
        from ouroboros.responses_executor import execute_streaming, ToolCallCapture
        from ouroboros.responses_translator import TranslatedInput

        capture = ToolCallCapture(chat_id=-1)
        # Pre-populate as if the bridge subscriber fired during the run.
        capture.on_tool_started("call_99", "read_file", {"path": "x.txt"})
        capture.on_tool_finished("call_99", "file body", is_error=False)

        async def _stub_bridge(*, chat_id, translated, timeout_sec, capture):
            return "done"

        monkeypatch.setattr(
            "ouroboros.responses_executor._run_through_bridge", _stub_bridge,
        )
        gen = execute_streaming(
            response_id="resp_2",
            model="openclaw",
            chat_id=-1,
            translated=TranslatedInput(user_text="go", system_prefix=""),
            capture=capture,
        )
        frames = self._consume(gen)
        joined = "".join(frames)
        # function_call sequence
        assert "response.output_item.added" in joined
        assert "response.function_call_arguments.delta" in joined
        assert "response.function_call_arguments.done" in joined
        # Custom tool_result item then assistant message.
        assert '"type": "tool_result"' in joined or "tool_result" in joined
        assert "response.completed" in joined

    def test_streaming_propagates_bridge_unavailable_as_failed(self, monkeypatch):
        from ouroboros.responses_executor import (
            execute_streaming, ToolCallCapture, BridgeUnavailable,
        )
        from ouroboros.responses_translator import TranslatedInput

        async def _stub_bridge(*, chat_id, translated, timeout_sec, capture):
            raise BridgeUnavailable("supervisor not ready yet")

        monkeypatch.setattr(
            "ouroboros.responses_executor._run_through_bridge", _stub_bridge,
        )
        gen = execute_streaming(
            response_id="resp_3", model="openclaw", chat_id=-1,
            translated=TranslatedInput(user_text="hi", system_prefix=""),
            capture=ToolCallCapture(chat_id=-1),
        )
        frames = self._consume(gen)
        joined = "".join(frames)
        assert "response.failed" in joined
        assert "supervisor not ready" in joined
        assert joined.rstrip().endswith("data: [DONE]")
