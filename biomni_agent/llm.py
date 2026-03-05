"""
LLM factory for Biomni on Databricks.

Uses Databricks Foundation Model API (OpenAI-compatible).
"""

import os
from typing import TYPE_CHECKING, Optional

from langchain_core.language_models.chat_models import BaseChatModel

if TYPE_CHECKING:
    from biomni_agent.config import BiomniConfig

# Default workspace URL for Foundation Model API (cluster provides auth)
DATABRICKS_HOST_ENV = "DATABRICKS_HOST"
# Endpoint base for pay-per-token: https://<workspace>.databricks.com/serving-endpoints/.../invocations
# LangChain OpenAI client with base_url works with: https://<workspace>.databricks.com/serving-endpoints/<endpoint>/invocations
# Actually the standard is: base_url = "https://<host>/serving-endpoints/<endpoint_name>/invocations"
# For OpenAI-compatible API on Databricks: base_url = f"https://{host}/serving-endpoints/{model}/invocations" and model = "" or the model id.
# See: https://docs.databricks.com/en/machine-learning/foundation-model-apis/index.html
# The recommended way is to use OpenAI client with base_url = f"https://{host}/serving-endpoints" and model = endpoint name.
# Or use databricks SDK / langchain-databricks. We use ChatOpenAI with base_url so it works on cluster with default auth.


def get_databricks_llm(
    model: str,
    temperature: float = 0.7,
    stop_sequences: list[str] | None = None,
    config: Optional["BiomniConfig"] = None,
) -> BaseChatModel:
    """
    Return a LangChain chat model that uses Databricks Foundation Model API.

    Uses the workspace's serving-endpoints API (OpenAI-compatible). On Databricks
    cluster, authentication is automatic. Set DATABRICKS_HOST if not on cluster.

    Args:
        model: Foundation Model API endpoint name (e.g. databricks-claude-sonnet-4-5).
        temperature: Sampling temperature.
        stop_sequences: Optional stop sequences.
        config: Optional config for overrides.
    """
    from langchain_openai import ChatOpenAI

    if config is not None:
        if model is None or model == "":
            model = config.llm
        if temperature is None:
            temperature = config.temperature

    if not model:
        model = "databricks-claude-sonnet-4-5"

    host = os.getenv(DATABRICKS_HOST_ENV)
    if not host:
        # On Databricks runtime, spark.conf has databricks URL; try env set by cluster
        try:
            import spark_connect
            spark_connect  # noqa: B018
        except ImportError:
            pass
        # Default: use relative URL when running on Databricks (same-origin)
        host = os.getenv("DATABRICKS_URL", "https://default.azuredatabricks.net")
    host = host.rstrip("/")
    # Foundation Model API OpenAI-compatible endpoint
    # https://docs.databricks.com/en/machine-learning/foundation-model-apis/api-reference.html
    base_url = f"{host}/serving-endpoints"

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.getenv("DATABRICKS_TOKEN", "placeholder"),  # Cluster auth may ignore this
        openai_api_base=base_url,
        stop=stop_sequences,
    )


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
    config: Optional["BiomniConfig"] = None,
) -> BaseChatModel:
    """
    Get LLM for Biomni agent. On Databricks uses Foundation Model API.
    """
    return get_databricks_llm(
        model=model or (config.llm if config else "databricks-claude-sonnet-4-5"),
        temperature=temperature or (config.temperature if config else 0.7),
        stop_sequences=stop_sequences,
        config=config,
    )
