"""Offline checks: progress visibility, retry diagnostics and unchanged responses."""
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

# Load the deployed client without creating a real OpenAI client or reading keys.
SOURCE = Path(__file__).resolve().parents[1] / "semantic_audit" / "api_client.py"
spec = importlib.util.spec_from_file_location("progress_client_test", SOURCE)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def make_client(tmp_path, create, retries=1):
    client = api.ParateraClient.__new__(api.ParateraClient)
    client.model = "offline-test"
    client.timeout = 60.0
    client.max_retries = retries
    client.calls = 0
    client.cache_dir = tmp_path / "cache"
    client.cache_dir.mkdir()
    client.heartbeat_seconds = 0.01
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client


def response():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Final Answer: 1"))], usage=None)


def test_wait_is_visible_before_request_finishes_and_stops_afterwards(tmp_path, capsys):
    def create(**kwargs):
        time.sleep(0.05)
        assert "等待API返回" in (tmp_path / "progress.log").read_text(encoding="utf-8")
        return response()

    client = make_client(tmp_path, create)
    result = client.chat([{"role": "user", "content": "PRIVATE_PROMPT"}])
    assert result["text"] == "Final Answer: 1"
    assert result["cached"] is False
    lines = (tmp_path / "progress.log").read_text(encoding="utf-8")
    time.sleep(0.03)
    assert (tmp_path / "progress.log").read_text(encoding="utf-8") == lines
    output = capsys.readouterr().out
    assert "发送API请求" in output and "API已返回" in output
    assert "PRIVATE_PROMPT" not in output


def test_failures_are_visible_without_printing_error_secrets(tmp_path, capsys, monkeypatch):
    def create(**kwargs):
        raise RuntimeError("PRIVATE_ERROR_SECRET")

    client = make_client(tmp_path, create, retries=2)
    monkeypatch.setattr(api.time, "sleep", lambda seconds: None)
    result = client.chat([{"role": "user", "content": "x"}])
    assert result["text"] == "" and client.calls == 2
    output = capsys.readouterr().out
    assert "type=RuntimeError" in output and "2/2" in output and "重试耗尽" in output
    assert "PRIVATE_ERROR_SECRET" not in output


def test_cache_hit_does_not_call_api(tmp_path, capsys):
    def create(**kwargs):
        pytest.fail("cache hit must not call API")

    client = make_client(tmp_path, create)
    messages = [{"role": "user", "content": "x"}]
    key = client._key(messages, 0.0, 1200)
    (client.cache_dir / (key + ".json")).write_text(json.dumps({"text": "cached answer"}), encoding="utf-8")
    result = client.chat(messages)
    assert result["cached"] is True and result["text"] == "cached answer" and client.calls == 0
    assert "缓存" in capsys.readouterr().out
