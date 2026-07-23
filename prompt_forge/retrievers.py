"""
Reference retriever implementations for use with InferenceAgent.context_retriever.

Each retriever is a callable ``(query: str, llm: LLMClient) -> str`` and can be
passed directly as a context_retriever or used as a starting point for custom logic.

Supported providers
-------------------
- ``"duckduckgo"``  — free, no API key (pip install "prompt-forge[duckduckgo]")
- ``"tavily"``      — best quality for RAG, free tier available
                      (pip install "prompt-forge[tavily]")

Optional: query rewriting
-------------------------
Set ``rewrite_query=True`` to have the LLM distil a focused search query from the
raw input before searching. Useful when inputs are long documents rather than
short natural-language queries.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM = (
    "You are a search query generator. "
    "Given an input text, output a single concise web search query (under 10 words) "
    "that would retrieve background information relevant to the input. "
    "Output the query only — no explanation, no quotes."
)


def _rewrite(query: str, llm: Any, max_input_chars: int = 1000) -> str:
    """Use the LLM to distil a focused search query from a potentially long input."""
    from .llm.client import LLMMessage

    try:
        resp = llm.complete([
            LLMMessage(role="system", content=_REWRITE_SYSTEM),
            LLMMessage(role="user", content=query[:max_input_chars]),
        ])
        rewritten = resp.text.strip()
        logger.debug("Query rewritten: %r → %r", query[:80], rewritten)
        return rewritten or query
    except Exception as exc:
        logger.warning("Query rewriting failed: %s — using raw query.", exc)
        return query


def _format_results(results: list[dict], *, title_key: str, url_key: str, snippet_key: str) -> str:
    """Render a list of search result dicts into a readable string."""
    if not results:
        return ""
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get(title_key, "")
        url = r.get(url_key, "")
        snippet = r.get(snippet_key, "")
        lines.append(f"[{i}] {title}\n{url}\n{snippet}")
    return "\n\n".join(lines)


class WebSearchRetriever:
    """
    Context retriever that fetches live web search results.

    Implements the retriever protocol ``(query: str, llm: LLMClient) -> str``
    so it can be passed directly as ``context_retriever``::

        from prompt_forge import Project
        from prompt_forge.retrievers import WebSearchRetriever

        retriever = WebSearchRetriever(provider="tavily", api_key="tvly-...")
        agent = project.get_inference_agent(context_retriever=retriever)

        # Same retriever in training so eval matches production
        report = project.train(
            train_bundles,
            val_bundles=val_bundles,
            config=TrainingConfig(context_retriever=retriever),
        )

    Args:
        provider:       Search backend — ``"duckduckgo"`` (free) or ``"tavily"``.
        api_key:        API key for providers that require one (Tavily).
        num_results:    Number of results to fetch and include in the context.
        rewrite_query:  If True, use the LLM to distil a focused search query
                        from the raw input before searching. Recommended when
                        inputs are long documents rather than short queries.
        max_input_chars: Maximum characters of the raw input passed to the query
                         rewriter (default 1000).
    """

    def __init__(
        self,
        provider: str = "duckduckgo",
        api_key: str | None = None,
        num_results: int = 3,
        rewrite_query: bool = False,
        max_input_chars: int = 1000,
    ):
        if provider not in ("duckduckgo", "tavily"):
            raise ValueError(f"Unknown provider {provider!r}. Choose 'duckduckgo' or 'tavily'.")
        self.provider = provider
        self.api_key = api_key
        self.num_results = num_results
        self.rewrite_query = rewrite_query
        self.max_input_chars = max_input_chars

    def __call__(self, query: str, llm: Any) -> str:
        if not query:
            return ""
        search_query = _rewrite(query, llm, self.max_input_chars) if self.rewrite_query else query
        if self.provider == "duckduckgo":
            return self._duckduckgo(search_query)
        return self._tavily(search_query)

    def _duckduckgo(self, query: str) -> str:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError(
                'duckduckgo-search is required for provider="duckduckgo". '
                'Install it with: pip install "prompt-forge[duckduckgo]"'
            )
        results = list(DDGS().text(query, max_results=self.num_results))
        return _format_results(results, title_key="title", url_key="href", snippet_key="body")

    def _tavily(self, query: str) -> str:
        if not self.api_key:
            raise ValueError("api_key is required for provider='tavily'.")
        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError(
                'tavily-python is required for provider="tavily". '
                'Install it with: pip install "prompt-forge[tavily]"'
            )
        client = TavilyClient(api_key=self.api_key)
        resp = client.search(query, max_results=self.num_results)
        results = resp.get("results", [])
        return _format_results(results, title_key="title", url_key="url", snippet_key="content")
