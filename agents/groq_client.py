"""Configuration commune et résiliente des clients Groq."""

import os

from langchain_groq import ChatGroq


DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"
DEFAULT_GROQ_MAX_RETRIES = 2


def _max_retries_from_env() -> int:
    raw_value = os.getenv(
        "GROQ_MAX_RETRIES",
        str(DEFAULT_GROQ_MAX_RETRIES),
    )

    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "GROQ_MAX_RETRIES doit être un entier positif ou nul."
        ) from exc

    if value < 0:
        raise ValueError(
            "GROQ_MAX_RETRIES doit être un entier positif ou nul."
        )

    return value


def create_groq_llm(*, max_tokens: int, temperature: float = 0 ,model: str | None = None):
    """Crée un client Groq avec retries exponentiels et modèle de secours.

    Le SDK Groq respecte ``retry-after`` lorsqu'il est fourni, puis applique
    un backoff exponentiel avec jitter. Une fois les tentatives du modèle
    principal épuisées, LangChain essaie le modèle de secours.
    """

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY manquante.")

    primary_model = (model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
).strip()
    fallback_model = os.getenv(
        "GROQ_FALLBACK_MODEL",
        DEFAULT_GROQ_FALLBACK_MODEL,
    ).strip()
    max_retries = _max_retries_from_env()

    common_options = {
        "temperature": temperature,
        "reasoning_effort": "low",
        "max_tokens": max_tokens,
        "api_key": api_key,
        "max_retries": max_retries,
    }

    primary = ChatGroq(
        model=primary_model,
        **common_options,
    )

    if not fallback_model or fallback_model == primary_model:
        return primary

    fallback = ChatGroq(
        model=fallback_model,
        **common_options,
    )

    return primary.with_fallbacks([fallback])
