import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    OPENAI_API_KEY: str
    OPENAI_EMBED_MODEL: str
    OPENAI_CHAT_MODEL: str
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    TOP_K: int
    SYSTEM_PROMPT: str

    @classmethod
    def from_env(cls):
        return cls(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
            OPENAI_EMBED_MODEL=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            OPENAI_CHAT_MODEL=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", "1000")),
            CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP", "200")),
            TOP_K=int(os.getenv("TOP_K", "4")),
            SYSTEM_PROMPT=os.getenv("SYSTEM_PROMPT", "You are a helpful assistant...")
        )
