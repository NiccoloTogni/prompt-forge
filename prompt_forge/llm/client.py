from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclasses.dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    text: str
    usage: dict[str, int] | None = None  # {"input_tokens": ..., "output_tokens": ...}
    raw: Any = None  # Original provider response for debugging


@dataclasses.dataclass
class TextPart:
    """A plain-text content part inside a multimodal message."""

    text: str
    type: str = "text"


@dataclasses.dataclass
class FilePart:
    """
    A file content part inside a multimodal message.

    The LLM client implementation decides how to deliver the file to its
    provider — base64 inline, pre-uploaded file ID, URL, etc.

    Args:
        path: Path to the file on disk.
        media_type: MIME type (e.g. "application/pdf", "image/png").
                    If None, the client may infer it from the file extension.
        file_id: Pre-uploaded file ID (e.g. from the Azure / OpenAI Files API).
                 When set, the client should use this instead of re-uploading.
    """

    path: Path
    media_type: str | None = None
    file_id: str | None = None
    type: str = "file"


# Type alias for message content — str for text-only, list for multimodal.
MessageContent = str | list[TextPart | FilePart]


@dataclasses.dataclass
class LLMMessage:
    """A single message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: MessageContent  # str for text-only, list[TextPart | FilePart] for multimodal


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol that any LLM client must satisfy.

    For text-only use cases, ``content`` is always a plain string and no
    special handling is needed. For native file support (PDFs, images, etc.),
    ``content`` may be a list of ``TextPart`` and ``FilePart`` objects — the
    client implementation is responsible for mapping these to its provider's
    multimodal format.

    Example: Azure OpenAI Responses API with native file support::

        import base64
        from openai import AzureOpenAI
        from prompt_forge import LLMMessage, LLMResponse, TextPart, FilePart

        class AzureClient:
            def __init__(self, deployment: str, **kwargs):
                self.client = AzureOpenAI(**kwargs)
                self.deployment = deployment

            def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
                input_ = []
                for m in messages:
                    if isinstance(m.content, str):
                        input_.append({"role": m.role, "content": m.content})
                    else:
                        parts = []
                        for part in m.content:
                            if isinstance(part, TextPart):
                                parts.append({"type": "input_text", "text": part.text})
                            elif isinstance(part, FilePart):
                                if part.file_id:
                                    parts.append({"type": "input_file", "file_id": part.file_id})
                                else:
                                    data = base64.b64encode(part.path.read_bytes()).decode()
                                    mime = part.media_type or "application/octet-stream"
                                    parts.append({
                                        "type": "input_file",
                                        "filename": part.path.name,
                                        "file_data": f"data:{mime};base64,{data}",
                                    })
                        input_.append({"role": m.role, "content": parts})

                resp = self.client.responses.create(model=self.deployment, input=input_, **kwargs)
                return LLMResponse(
                    text=resp.output_text,
                    usage={"input_tokens": resp.usage.input_tokens,
                           "output_tokens": resp.usage.output_tokens},
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
            messages: List of LLMMessage objects. Message content is either a
                      plain string (text-only) or a list of TextPart / FilePart
                      objects (multimodal). Clients that do not support native
                      file inputs should raise a clear error if they receive a
                      FilePart in the content.
            **kwargs: Provider-specific options (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with at minimum the .text field populated.
        """
        ...
