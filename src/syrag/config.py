from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ComponentDefaults(BaseModel):
    embedder: str | None = Field(default=None)
    vector_store: str | None = Field(default=None)
    llm: str | None = Field(default=None)


class InMemoryProviderSettings(BaseModel):
    embedder_dimensions: int = Field(default=16, gt=0)
    llm_max_context_documents: int = Field(default=3, gt=0)


class ProviderSettings(BaseModel):
    in_memory: InMemoryProviderSettings = Field(default_factory=InMemoryProviderSettings)


class BootstrapSettings(BaseModel):
    register_in_memory_defaults: bool = Field(default=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SYRAG_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = Field(default="SyRAG")
    app_version: str = Field(default="0.2.0")
    environment: str = Field(default="development")
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=True)
    defaults: ComponentDefaults = Field(default_factory=ComponentDefaults)
    bootstrap: BootstrapSettings = Field(default_factory=BootstrapSettings)
    providers: ProviderSettings = Field(default_factory=ProviderSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
