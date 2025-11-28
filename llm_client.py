# llm_client.py (only this part changes)
import os
from typing import Optional, List, Dict, Any
from openai import OpenAI
import re

HF_TOKEN = os.getenv("HF_TOKEN")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "HuggingFaceTB/SmolLM3-3B:hf-inference")
HF_ROUTER_URL = os.getenv("HF_ROUTER_URL", "https://router.huggingface.co/v1")

if HF_TOKEN is None:
    raise RuntimeError(
        "HF_TOKEN environment variable is not set. "
        "Set HF_TOKEN to your Hugging Face access token."
    )

client = OpenAI(
    base_url=HF_ROUTER_URL,
    api_key=HF_TOKEN,
)


def _strip_think_tags(text: str) -> str:
    """
    Remove `<think>...</think>` blocks from the LLM output,
    keep only the visible final content.
    """
    # Remove all <think>...</think> segments
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Strip extra whitespace
    return cleaned.strip()


def call_llm(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 512,
) -> str:
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model=HF_LLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
    )

    msg = completion.choices[0].message
    content = msg.content or ""
    content = _strip_think_tags(content)
    return content