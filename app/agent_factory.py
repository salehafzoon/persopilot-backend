# app/agent_factory.py
from functools import lru_cache
from src.backend.agents.agent import PersoAgent

MODEL_PATH = "LLMs/phi-4-mini-instruct"
TOKENIZER_PATH = "Tokenizers/phi-4-mini-instruct"

@lru_cache  # Singleton
def get_agent() -> PersoAgent:
    """Return a global PersoAgent instance (loaded once)."""
    return PersoAgent(model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH)
