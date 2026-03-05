import dataclasses
from typing import Any, Protocol, runtime_checkable


@dataclasses.dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    text: str
    usage: dict[str, int] | None = None  # {"input_tokens": ..., "output_tokens": ...}
    raw: Any = None  # Original provider response for debugging


@dataclasses.dataclass
class LLMMessage:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str | list[dict[str, Any]]  # str for text, list for multimodal


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol that any LLM client must satisfy.

    Example implementation for Azure OpenAI:

        from openai import AzureOpenAI

        class AzureClient:
            def __init__(self, deployment: str, **kwargs):
                self.client = AzureOpenAI(**kwargs)
                self.deployment = deployment

            def complete(self, messages, **kwargs) -> LLMResponse:
                resp = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                    **kwargs,
                )
                return LLMResponse(
                    text=resp.choices[0].message.content,
                    usage={"input_tokens": resp.usage.prompt_tokens,
                           "output_tokens": resp.usage.completion_tokens},
                    raw=resp,
                )
    """

    def complete(
        self,
        messages: list[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send messages to the LLM and return a response.

        Args:
            messages: List of LLMMessage objects (system, user, assistant).
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with at minimum the .text field populated.
        """
        ...
