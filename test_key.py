"""Smoke test: confirms the API key loads and the Claude API responds."""
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()  

key = os.getenv("ANTHROPIC_API_KEY")
if not key:
    raise SystemExit("ANTHROPIC_API_KEY not found. Check .env is in this folder and saved.")

client = anthropic.Anthropic(api_key=key)
resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=50,
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(resp.content[0].text)
