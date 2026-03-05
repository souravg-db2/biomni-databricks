"""
LLM factory for Biomni on Databricks.

Uses ChatDatabricks for native Databricks Foundation Model API support.
Tries langchain_databricks first, then langchain_community, then falls
back to a patched ChatOpenAI.
"""

import json
import os
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

if TYPE_CHECKING:
    from biomni_agent.config import BiomniConfig


def _get_chat_databricks_class():
    """Find the best available ChatDatabricks implementation."""
    try:
        from langchain_databricks import ChatDatabricks
        return ChatDatabricks
    except ImportError:
        pass
    try:
        from langchain_community.chat_models import ChatDatabricks
        return ChatDatabricks
    except ImportError:
        pass
    return None


def get_databricks_llm(
    model: str,
    temperature: float = 0.7,
    stop_sequences: list[str] | None = None,
    config: Optional["BiomniConfig"] = None,
) -> BaseChatModel:
    """Return a LangChain chat model using Databricks Foundation Model API."""
    if config is not None:
        if model is None or model == "":
            model = config.llm
        if temperature is None:
            temperature = config.temperature

    if not model:
        model = "databricks-claude-sonnet-4-5"

    ChatDatabricks = _get_chat_databricks_class()
    if ChatDatabricks is not None:
        return ChatDatabricks(
            endpoint=model,
            temperature=temperature,
            stop=stop_sequences,
        )

    from langchain_openai import ChatOpenAI

    class _PatchedChatOpenAI(ChatOpenAI):
        """Fallback: ChatOpenAI with patched response handling for Databricks."""

        def _create_chat_result(self, response: Any, generation_info: dict | None = None) -> Any:
            if isinstance(response, str):
                response = json.loads(response)
            return super()._create_chat_result(response, generation_info)

    host = os.getenv("DATABRICKS_HOST")
    if not host:
        host = os.getenv("DATABRICKS_URL", "https://default.azuredatabricks.net")
    base_url = f"{host.rstrip('/')}/serving-endpoints"

    return _PatchedChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.getenv("DATABRICKS_TOKEN", "placeholder"),
        openai_api_base=base_url,
        stop=stop_sequences,
    )


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
    config: Optional["BiomniConfig"] = None,
) -> BaseChatModel:
    """Get LLM for Biomni agent. On Databricks uses Foundation Model API."""
    return get_databricks_llm(
        model=model or (config.llm if config else "databricks-claude-sonnet-4-5"),
        temperature=temperature or (config.temperature if config else 0.7),
        stop_sequences=stop_sequences,
        config=config,
    )
