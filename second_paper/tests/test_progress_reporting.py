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
    with pytest.raises(RuntimeError, match="failed after 2 attempts") as error:
        client.chat([{"role": "user", "content": "x"}])
    assert "PRIVATE_ERROR_SECRET" not in str(error.value)
    assert client.calls == 2
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


def test_wtq_runner_writes_sample_progress_with_offline_model(tmp_path, capsys, monkeypatch):
    import sys

    runner_path = SOURCE.parents[1] / "scripts" / "run_paper1.py"
    runner_spec = importlib.util.spec_from_file_location("offline_progress_runner", runner_path)
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)

    def fake_client(cache_dir, timeout):
        def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="```python\ndef solve(df):\n    return 1\n```"))], usage=None)
        return make_client(tmp_path, create)

    monkeypatch.setattr(runner, "ParateraClient", fake_client)
    output = tmp_path / "offline_wtq"
    monkeypatch.setattr(sys, "argv", ["run_paper1.py", "--dataset", "wtq", "--mode", "baseline",
                                   "--start", "0", "--end", "1", "--all-questions",
                                   "--experiment", str(output)])
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == 0
    console = capsys.readouterr().out
    assert "样本1/1" in console and "完成 1/1 (100%)" in console
    assert "预计剩余=" in console
    assert (output / "log" / "0.txt").is_file()
    assert len((output / "result.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_glm_retries_truncation_once_and_counts_both_responses(tmp_path, monkeypatch):
    monkeypatch.setenv("PARATERA_GLM_THINKING", "disabled")
    budgets = []

    def create(**kwargs):
        budgets.append(kwargs["max_tokens"])
        truncated = len(budgets) == 1
        from contextlib import nullcontext
        assert kwargs["stream"] and kwargs["stream_options"]["include_usage"]
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        choice = SimpleNamespace(delta=SimpleNamespace(content="" if truncated else "answer = 1"),
                                 finish_reason="length" if truncated else "stop")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=100, total_tokens=110)
        hidden = SimpleNamespace(delta=SimpleNamespace(content=None, reasoning_content="private"), finish_reason=None)
        chunks = [SimpleNamespace(choices=[hidden]), SimpleNamespace(choices=[choice]),
                  SimpleNamespace(choices=[], usage=usage)]
        return nullcontext(iter(chunks))

    client = make_client(tmp_path, create, retries=3)
    client.model = "GLM-5.3"
    result = client.chat([{"role": "user", "content": "test"}], max_tokens=4096)
    assert budgets == [4096, 8192]
    assert result["text"] == "answer = 1"
    assert result["api_calls"] == 2 and result["usage"]["total_tokens"] == 220
    monkeypatch.setenv("PARATERA_GLM_THINKING", "default")
    assert client._key([{"role": "user", "content": "test"}], 0.0, 4096) != result["prompt_hash"]


def test_empty_cache_and_empty_response_do_not_reach_executor(tmp_path):
    messages = [{"role": "user", "content": "test"}]
    empty = SimpleNamespace(message=SimpleNamespace(content=" "), finish_reason="stop")
    client = make_client(tmp_path, lambda **kwargs: SimpleNamespace(choices=[empty], usage=None))
    key = client._key(messages, 0.0, 1200)
    path = client.cache_dir / (key + ".json")
    path.write_text(json.dumps({"text": ""}), encoding="utf-8")
    with pytest.raises(api.ModelOutputError):
        client.chat(messages)
    assert client.calls == 1


def test_empty_python_is_an_execution_failure():
    import pandas as pd
    from second_paper.semantic_audit.runtime_trace import execute_code

    for snippet in ("", "  ", "# no program"):
        result = execute_code(snippet, pd.DataFrame())
        assert not result["success"] and "EMPTY_GENERATED_CODE" in result["error"]
