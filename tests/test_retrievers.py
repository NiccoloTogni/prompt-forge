"""
Tests for WebSearchRetriever.
Provider calls are fully mocked — no network access required.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from prompt_forge.retrievers import WebSearchRetriever, _rewrite, _format_results
from prompt_forge.llm.client import LLMMessage, LLMResponse


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_llm(rewrite_text: str = "focused query") -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(text=rewrite_text, usage={})
    return llm


DDG_RESULTS = [
    {"title": "Result 1", "href": "https://example.com/1", "body": "snippet one"},
    {"title": "Result 2", "href": "https://example.com/2", "body": "snippet two"},
]

TAVILY_RESPONSE = {
    "results": [
        {"title": "T1", "url": "https://tavily.com/1", "content": "tavily snippet one"},
        {"title": "T2", "url": "https://tavily.com/2", "content": "tavily snippet two"},
    ]
}


def _mock_ddg(results=DDG_RESULTS):
    """Patch duckduckgo_search so no real network call is made."""
    ddg_mod = ModuleType("duckduckgo_search")
    ddgs_instance = MagicMock()
    ddgs_instance.text.return_value = results
    ddg_mod.DDGS = MagicMock(return_value=ddgs_instance)
    return patch.dict(sys.modules, {"duckduckgo_search": ddg_mod}), ddgs_instance


def _mock_tavily(response=TAVILY_RESPONSE):
    tavily_mod = ModuleType("tavily")
    client_instance = MagicMock()
    client_instance.search.return_value = response
    tavily_mod.TavilyClient = MagicMock(return_value=client_instance)
    return patch.dict(sys.modules, {"tavily": tavily_mod}), client_instance


# ── Constructor validation ────────────────────────────────────────────────────

def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        WebSearchRetriever(provider="bing")


def test_tavily_without_api_key_raises():
    patcher, _ = _mock_tavily()
    with patcher:
        r = WebSearchRetriever(provider="tavily")
        with pytest.raises(ValueError, match="api_key"):
            r("query", MagicMock())


# ── DuckDuckGo ────────────────────────────────────────────────────────────────

def test_duckduckgo_returns_formatted_results():
    patcher, ddgs = _mock_ddg()
    with patcher:
        r = WebSearchRetriever(provider="duckduckgo", num_results=2)
        result = r("python tips", MagicMock())
    assert "Result 1" in result
    assert "https://example.com/1" in result
    assert "snippet one" in result


def test_duckduckgo_passes_num_results():
    patcher, ddgs = _mock_ddg()
    with patcher:
        r = WebSearchRetriever(provider="duckduckgo", num_results=5)
        r("query", MagicMock())
    ddgs.text.assert_called_once_with("query", max_results=5)


def test_duckduckgo_empty_results_returns_empty_string():
    patcher, _ = _mock_ddg(results=[])
    with patcher:
        r = WebSearchRetriever(provider="duckduckgo")
        assert r("query", MagicMock()) == ""


def test_duckduckgo_missing_package_raises_import_error():
    with patch.dict(sys.modules, {"duckduckgo_search": None}):
        r = WebSearchRetriever(provider="duckduckgo")
        with pytest.raises(ImportError, match="duckduckgo-search"):
            r("query", MagicMock())


# ── Tavily ────────────────────────────────────────────────────────────────────

def test_tavily_returns_formatted_results():
    patcher, _ = _mock_tavily()
    with patcher:
        r = WebSearchRetriever(provider="tavily", api_key="test-key")
        result = r("LLM tips", MagicMock())
    assert "T1" in result
    assert "https://tavily.com/1" in result
    assert "tavily snippet one" in result


def test_tavily_passes_api_key():
    patcher, client = _mock_tavily()
    with patcher:
        from tavily import TavilyClient
        r = WebSearchRetriever(provider="tavily", api_key="my-key")
        r("query", MagicMock())
    TavilyClient.assert_called_once_with(api_key="my-key")


def test_tavily_missing_package_raises_import_error():
    with patch.dict(sys.modules, {"tavily": None}):
        r = WebSearchRetriever(provider="tavily", api_key="key")
        with pytest.raises(ImportError, match="tavily-python"):
            r("query", MagicMock())


# ── Query rewriting ───────────────────────────────────────────────────────────

def test_rewrite_query_calls_llm():
    patcher, ddgs = _mock_ddg()
    llm = make_llm("focused query")
    with patcher:
        r = WebSearchRetriever(provider="duckduckgo", rewrite_query=True)
        r("a very long document text...", llm)
    ddgs.text.assert_called_once_with("focused query", max_results=r.num_results)


def test_rewrite_query_false_passes_raw_query():
    patcher, ddgs = _mock_ddg()
    llm = make_llm()
    with patcher:
        r = WebSearchRetriever(provider="duckduckgo", rewrite_query=False)
        r("raw query", llm)
    ddgs.text.assert_called_once_with("raw query", max_results=r.num_results)
    llm.complete.assert_not_called()


def test_rewrite_truncates_long_input():
    received = []
    def capture(msgs, **kw):
        received.append(msgs[1].content)
        return LLMResponse(text="q", usage={})
    llm = MagicMock()
    llm.complete.side_effect = capture

    long_input = "x" * 5000
    patcher, _ = _mock_ddg()
    with patcher:
        r = WebSearchRetriever(provider="duckduckgo", rewrite_query=True, max_input_chars=100)
        r(long_input, llm)
    assert len(received[0]) == 100


def test_rewrite_fallback_on_llm_error():
    patcher, ddgs = _mock_ddg()
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("llm down")
    with patcher:
        r = WebSearchRetriever(provider="duckduckgo", rewrite_query=True)
        r("original query", llm)
    # Falls back to original query rather than raising
    ddgs.text.assert_called_once_with("original query", max_results=r.num_results)


# ── Empty query ───────────────────────────────────────────────────────────────

def test_empty_query_returns_empty_string():
    r = WebSearchRetriever(provider="duckduckgo")
    assert r("", MagicMock()) == ""


# ── _format_results ───────────────────────────────────────────────────────────

def test_format_results_structure():
    results = [{"t": "Title", "u": "https://x.com", "s": "snip"}]
    out = _format_results(results, title_key="t", url_key="u", snippet_key="s")
    assert "[1] Title" in out
    assert "https://x.com" in out
    assert "snip" in out


def test_format_results_empty():
    assert _format_results([], title_key="t", url_key="u", snippet_key="s") == ""


# ── Public export ─────────────────────────────────────────────────────────────

def test_web_search_retriever_is_exported():
    import prompt_forge
    assert hasattr(prompt_forge, "WebSearchRetriever")
