"""Base class for all EPAC role agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class RoleConfig(BaseModel):
    """Configuration for a single EPAC role agent."""

    role_name: str
    system_prompt: str = ""
    llm_model: str = "openai/gpt-4o"
    llm_temperature: float = 0.2
    max_tokens: int = 16000
    tools: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    # Prompt size controls (0 = unlimited)
    max_file_content_chars: int = 20000
    max_actor_feedback_chars: int = 10000
    max_actor_feedback_findings: int = 50


class BaseRole(ABC):
    """
    Abstract base class for all EPAC role agents.

    Each role receives the shared EPACState, performs its work, and returns
    an updated state dict.  Roles must NOT access other roles' state directly —
    they communicate only through typed artifacts.
    """

    def __init__(self, config: RoleConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.role_name

    @abstractmethod
    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute this role's logic.

        Parameters
        ----------
        state:
            The current EPACState as a dict (LangGraph passes state as dicts).

        Returns
        -------
        dict
            Partial state update dict that LangGraph will merge into the shared state.
        """

    def _build_messages(self, user_content: str) -> list[dict[str, str]]:
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages
