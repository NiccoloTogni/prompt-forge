"""Build an LLM client from app config."""

import streamlit as st
from openai import AzureOpenAI
from prompt_forge import LLMResponse

from prompt_forge.app.state import load_app_config


def build_llm():
    """Instantiate an AzureOpenAI-backed LLM client from current config.

    Returns None if credentials are missing.
    """
    cfg = load_app_config()
    if not cfg.get("azure_api_key") or not cfg.get("azure_endpoint"):
        return None

    try:
        class _AzureClient:
            def __init__(self):
                self._client = AzureOpenAI(
                    api_version=cfg["azure_api_version"],
                    azure_endpoint=cfg["azure_endpoint"],
                    api_key=cfg["azure_api_key"],
                )
                self._deployment = cfg["azure_deployment"]

            def complete(self, messages: list, **kwargs) -> LLMResponse:
                allowed = {"temperature", "max_tokens", "top_p",
                           "frequency_penalty", "presence_penalty"}
                resp = self._client.chat.completions.create(
                    model=self._deployment,
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                    **{k: v for k, v in kwargs.items() if k in allowed},
                )
                return LLMResponse(
                    text=resp.choices[0].message.content,
                    usage={
                        "input_tokens": resp.usage.prompt_tokens,
                        "output_tokens": resp.usage.completion_tokens,
                    },
                )

        return _AzureClient()

    except Exception as e:
        st.error(f"Failed to build LLM client: {e}")
        return None


def get_or_build_llm():
    """Return cached LLM from session_state, rebuilding if absent."""
    if st.session_state.get("llm") is None:
        st.session_state.llm = build_llm()
    return st.session_state.llm
