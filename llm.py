
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Default: small, inexpensive model.
MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")

_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def call_llm(messages: list, tools: list | None = None,
             system: str | None = None, max_tokens: int = 1024):
    """Send one request to Claude and return the raw response object."""
    kwargs = {"model": MODEL, "max_tokens": max_tokens, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    if system:
        kwargs["system"] = system
    return _client.messages.create(**kwargs)