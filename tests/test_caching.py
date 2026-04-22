"""
Tests for CachedLLM.
"""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prompt_forge.caching import CachedLLM, _make_key
from prompt_forge.llm.client import LLMMessage, LLMResponse, TextPart, FilePart


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_llm(text: str = "answer") -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(
        text=text, usage={"input_tokens": 10, "output_tokens": 5}
    )
    return llm


def text_messages(content: str = "hi") -> list[LLMMessage]:
    return [
        LLMMessage(role="system", content="sys"),
        LLMMessage(role="user", content=content),
    ]


# ── Basic hit / miss ──────────────────────────────────────────────────────────

def test_cache_miss_calls_underlying_llm():
    llm = make_llm("answer")
    cached = CachedLLM(llm)
    result = cached.complete(text_messages())
    assert result.text == "answer"
    assert llm.complete.call_count == 1


def test_cache_hit_does_not_call_underlying_llm():
    llm = make_llm("answer")
    cached = CachedLLM(llm)
    cached.complete(text_messages())
    cached.complete(text_messages())
    assert llm.complete.call_count == 1


def test_cache_hit_returns_same_text():
    llm = make_llm("answer")
    cached = CachedLLM(llm)
    r1 = cached.complete(text_messages())
    r2 = cached.complete(text_messages())
    assert r1.text == r2.text == "answer"


def test_cache_hit_reports_zero_tokens():
    llm = make_llm("answer")
    cached = CachedLLM(llm)
    cached.complete(text_messages())
    r2 = cached.complete(text_messages())
    assert r2.usage == {"input_tokens": 0, "output_tokens": 0}


def test_different_inputs_both_miss():
    llm = make_llm()
    cached = CachedLLM(llm)
    cached.complete(text_messages("hello"))
    cached.complete(text_messages("world"))
    assert llm.complete.call_count == 2


def test_kwargs_are_part_of_key():
    llm = make_llm()
    cached = CachedLLM(llm)
    msgs = text_messages()
    cached.complete(msgs, temperature=0.0)
    cached.complete(msgs, temperature=1.0)
    assert llm.complete.call_count == 2


# ── Hit rate ──────────────────────────────────────────────────────────────────

def test_hit_rate_none_before_any_calls():
    cached = CachedLLM(make_llm())
    assert cached.hit_rate is None


def test_hit_rate_calculation():
    llm = make_llm()
    cached = CachedLLM(llm)
    cached.complete(text_messages("a"))   # miss
    cached.complete(text_messages("a"))   # hit
    cached.complete(text_messages("a"))   # hit
    assert cached.hits == 2
    assert cached.misses == 1
    assert abs(cached.hit_rate - 2 / 3) < 1e-9


# ── External cache store ──────────────────────────────────────────────────────

def test_external_cache_is_populated():
    store = {}
    llm = make_llm("stored")
    cached = CachedLLM(llm, cache=store)
    cached.complete(text_messages())
    assert len(store) == 1


def test_external_cache_is_reused_across_instances():
    store = {}
    llm = make_llm("reused")
    CachedLLM(llm, cache=store).complete(text_messages())
    # Second instance with the same store should hit the cache
    cached2 = CachedLLM(llm, cache=store)
    r = cached2.complete(text_messages())
    assert r.text == "reused"
    assert llm.complete.call_count == 1


# ── File part cache key ───────────────────────────────────────────────────────

def test_file_part_key_uses_content_not_path(tmp_path):
    f1 = tmp_path / "a.pdf"
    f2 = tmp_path / "b.pdf"   # different path, same content
    f1.write_bytes(b"%PDF-same")
    f2.write_bytes(b"%PDF-same")

    msgs1 = [LLMMessage(role="user", content=[FilePart(path=f1, media_type="application/pdf")])]
    msgs2 = [LLMMessage(role="user", content=[FilePart(path=f2, media_type="application/pdf")])]
    assert _make_key(msgs1, {}) == _make_key(msgs2, {})


def test_file_part_key_differs_when_content_differs(tmp_path):
    f1 = tmp_path / "a.pdf"
    f2 = tmp_path / "b.pdf"
    f1.write_bytes(b"%PDF-v1")
    f2.write_bytes(b"%PDF-v2")

    msgs1 = [LLMMessage(role="user", content=[FilePart(path=f1, media_type="application/pdf")])]
    msgs2 = [LLMMessage(role="user", content=[FilePart(path=f2, media_type="application/pdf")])]
    assert _make_key(msgs1, {}) != _make_key(msgs2, {})


def test_file_part_cache_hit(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF")
    msgs = [LLMMessage(role="user", content=[FilePart(path=f, media_type="application/pdf")])]

    llm = make_llm("file answer")
    cached = CachedLLM(llm)
    cached.complete(msgs)
    r = cached.complete(msgs)
    assert r.text == "file answer"
    assert llm.complete.call_count == 1


# ── Protocol compatibility ────────────────────────────────────────────────────

def test_cached_llm_is_accepted_as_llm_client():
    from prompt_forge.inference.agent import InferenceAgent
    llm = make_llm("ok")
    cached = CachedLLM(llm)
    agent = InferenceAgent(llm=cached, prompt_text="sys", native_files=False)
    result = agent.run(input_text="hello")
    assert result == "ok"
    # Second call is a cache hit
    agent.run(input_text="hello")
    assert llm.complete.call_count == 1
