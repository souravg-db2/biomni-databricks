"""
Biomni Databricks configuration.

Uses UC Volume for data path; no S3 download at agent init.
"""

import os
from dataclasses import dataclass


# Default UC Volume path for data lake (plan: /Volumes/biomni/agent/raw_files)
DEFAULT_DATA_LAKE_VOLUME_PATH = "/Volumes/biomni/agent/raw_files"


@dataclass
class BiomniConfig:
    """Configuration for Biomni agent on Databricks."""

    # Data path: UC Volume (no S3 download)
    path: str = DEFAULT_DATA_LAKE_VOLUME_PATH
    timeout_seconds: int = 600

    # LLM: Databricks Foundation Model API endpoint name
    llm: str = "databricks-claude-sonnet-4-5"
    temperature: float = 0.7

    use_tool_retriever: bool = True
    commercial_mode: bool = False

    # Optional: custom model (not used when using Databricks FM API)
    base_url: str | None = None
    api_key: str | None = None
    source: str | None = None  # "Databricks" when using FM API

    def __post_init__(self):
        if os.getenv("BIOMNI_PATH"):
            self.path = os.getenv("BIOMNI_PATH")
        if os.getenv("BIOMNI_DATA_LAKE_PATH"):
            self.path = os.getenv("BIOMNI_DATA_LAKE_PATH")
        if os.getenv("BIOMNI_TIMEOUT_SECONDS"):
            self.timeout_seconds = int(os.getenv("BIOMNI_TIMEOUT_SECONDS"))
        if os.getenv("BIOMNI_LLM"):
            self.llm = os.getenv("BIOMNI_LLM")
        if os.getenv("BIOMNI_USE_TOOL_RETRIEVER"):
            self.use_tool_retriever = os.getenv("BIOMNI_USE_TOOL_RETRIEVER", "").lower() == "true"
        if os.getenv("BIOMNI_COMMERCIAL_MODE"):
            self.commercial_mode = os.getenv("BIOMNI_COMMERCIAL_MODE", "").lower() == "true"
        if os.getenv("BIOMNI_TEMPERATURE"):
            self.temperature = float(os.getenv("BIOMNI_TEMPERATURE"))

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "timeout_seconds": self.timeout_seconds,
            "llm": self.llm,
            "temperature": self.temperature,
            "use_tool_retriever": self.use_tool_retriever,
            "commercial_mode": self.commercial_mode,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "source": self.source,
        }


default_config = BiomniConfig()
