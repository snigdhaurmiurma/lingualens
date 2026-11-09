"""Schemas for tokenization inspector endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TokenizationInspectRequest(BaseModel):
    """Payload to inspect subword token fragmentation."""

    text: str = Field(min_length=1, max_length=2000, description="Surface text to tokenize")
    model_id: str = Field(default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", description="Target model ID or alias")
    language_code: str = Field(default="unknown", description="ISO language code of input text")


class TokenizationInspectResponse(BaseModel):
    """Tokenization inspection result schema."""

    model_id: str
    language_code: str
    text: str
    tokens: list[str]
    token_count: int
    text_tokens: list[str]
    text_token_count: int
    word_count: int
    subword_fertility_ratio: float
    has_unknown_tokens: bool
    unknown_token_count: int
    limitations_note: str
