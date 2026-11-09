"""Tokenization inspector API endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.adapters.registry import default_registry
from app.schemas.tokenization import TokenizationInspectRequest, TokenizationInspectResponse
from app.services.tokenizer_service import TokenizerInspector

router = APIRouter(prefix="/tokenization", tags=["Tokenization"])


@router.post("/inspect", response_model=TokenizationInspectResponse, status_code=status.HTTP_200_OK)
def inspect_tokenization(payload: TokenizationInspectRequest) -> TokenizationInspectResponse:
    """Inspect subword tokenization, sequence boundary tokens, and subword fertility."""
    try:
        adapter = default_registry.get_adapter(payload.model_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Embedding model '{payload.model_id}' is not registered.",
        )

    inspector = TokenizerInspector(adapter=adapter)
    result = inspector.inspect_text(
        text=payload.text,
        language_code=payload.language_code,
    )

    limitations = TokenizerInspector.get_methodological_limitations()
    limitations_note = "; ".join(f"{k}: {v}" for k, v in limitations.items())

    return TokenizationInspectResponse(
        model_id=adapter.model_id,
        language_code=result.language_code,
        text=result.text,
        tokens=result.tokens,
        token_count=result.token_count,
        text_tokens=result.text_tokens,
        text_token_count=result.text_token_count,
        word_count=result.word_count,
        subword_fertility_ratio=result.subword_fertility_ratio,
        has_unknown_tokens=result.has_unknown_tokens,
        unknown_token_count=result.unknown_token_count,
        limitations_note=limitations_note,
    )

