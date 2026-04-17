from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FASTRAG_",
        extra="ignore",
    )

    app_name: str = Field(default="FastRAG")
    app_version: str = Field(default="0.1.0")
    environment: str = Field(default="development")
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
