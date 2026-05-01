"""
Internal LLM dispatch layer.

Uses LiteLLM for model-agnostic calls.  Supports any provider via the
standard `provider/model` format (e.g. "openai/gpt-4o", "anthropic/claude-3-7-sonnet-20250219").

Users can override this module entirely by subclassing BaseRole and overriding
_call_llm_for_* methods, or by pointing EPAC_LLM_PROVIDER env vars at a custom endpoint.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_LITELLM_AVAILABLE = False
try:
    import litellm  # type: ignore[import]
    _LITELLM_AVAILABLE = True
except ImportError:
    pass


async def call_llm_json(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 8192,
) -> dict[str, Any]:
    """
    Call an LLM and return the response parsed as JSON.

    Parameters
    ----------
    model:
        LiteLLM-style model string, e.g. "openai/gpt-4o"
    system_prompt:
        Role system prompt
    user_prompt:
        Task-specific prompt
    temperature:
        LLM temperature
    max_tokens:
        Maximum response tokens

    Returns
    -------
    dict
        Parsed JSON response from the LLM
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if _LITELLM_AVAILABLE:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content or "{}"
    else:
        # Fallback: if litellm isn't installed, attempt direct OpenAI call
        try:
            from openai import AsyncOpenAI  # type: ignore[import]

            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            # Strip provider prefix for direct OpenAI calls
            openai_model = model.split("/", 1)[-1]
            resp = await client.chat.completions.create(
                model=openai_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw_text = resp.choices[0].message.content or "{}"
        except Exception as exc:
            raise RuntimeError(
                "No LLM client available. Install litellm (`pip install epac[litellm]`) "
                "or openai (`pip install epac[openai]`)."
            ) from exc

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        # Try to extract JSON block from markdown-wrapped response
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw_text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as inner_exc:
                raise ValueError(f"LLM did not return valid JSON: {raw_text[:500]}") from inner_exc
        raise ValueError(f"LLM did not return valid JSON: {raw_text[:500]}") from exc
