"""Application configuration using Pydantic Settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Analyst + Auditor Research Agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_adversarial_model: str = "gemini-2.5-flash"

    # Tavily Search API
    tavily_api_key: str = ""

    # Hard constraints & budgets per question
    max_time_seconds: float = 120.0
    max_search_calls: int = 10
    max_fetch_calls: int = 10
    max_llm_calls: int = 12
    max_input_tokens: int = 50000
    max_output_tokens: int = 10000

    # Sub-operation timeouts (seconds)
    search_timeout_seconds: float = 15.0
    fetch_timeout_seconds: float = 15.0
    llm_timeout_seconds: float = 30.0

    # Cost accounting constants
    # Default Gemini 2.5 Flash pricing: $0.15 / 1M prompt tokens, $0.60 / 1M completion tokens
    cost_per_million_input_tokens: float = 0.15
    cost_per_million_output_tokens: float = 0.60
    tavily_cost_per_search: float = 0.005  # Tavily approximate API cost per query
    tavily_cost_per_extract: float = 0.005

    # Currency conversion
    usd_to_inr_rate: float = 86.50
    usd_to_inr_date: str = "2026-09-26"

    # Storage and logs directories
    database_path: str = "data/research.db"
    logs_dir: str = "logs"
    results_dir: str = "results"

    @property
    def has_valid_gemini_key(self) -> bool:
        """Check if a genuine Gemini API key is configured."""
        val = self.gemini_api_key.strip() if self.gemini_api_key else ""
        return bool(val and val != "your_gemini_api_key_here")

    @property
    def has_valid_tavily_key(self) -> bool:
        """Check if a genuine Tavily API key is configured."""
        val = self.tavily_api_key.strip() if self.tavily_api_key else ""
        return bool(val and val != "your_tavily_api_key_here")

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)


# Singleton settings instance
settings = Settings()
