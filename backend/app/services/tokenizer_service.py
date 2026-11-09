"""Tokenizer Diagnostic Service & Inspector.

Provides cross-lingual subword fragmentation analysis across:
- English (en)
- Bengali (bn)
- Banglish / Bengali Transliteration (bn-Latn)
- Hindi (hi)
- Arabic (ar)

Includes scientific metrics (subword fertility ratio) and documents diagnostic limitations.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from app.adapters.base import BaseEmbeddingAdapter, TokenizationResult


class TokenizerInspector:
    """Diagnostic service auditing subword tokenization across languages and scripts."""

    def __init__(self, adapter: BaseEmbeddingAdapter) -> None:
        self.adapter = adapter

    def inspect_text(self, text: str, language_code: str = "unknown") -> TokenizationResult:
        """Inspect a single text string and calculate subword fragmentation metrics."""
        return self.adapter.tokenize(text, language_code=language_code)

    def inspect_multilingual_query_group(
        self,
        query_variants: Mapping[str, str],
    ) -> dict[str, TokenizationResult]:
        """Inspect equivalent query formulations across multiple languages.

        Args:
            query_variants: Mapping of language codes (e.g. 'en', 'bn', 'bn-Latn')
                            to query text strings.

        Returns:
            Dictionary mapping language codes to TokenizationResult objects.
        """
        results: dict[str, TokenizationResult] = {}
        for lang_code, text in query_variants.items():
            results[lang_code] = self.inspect_text(text, language_code=lang_code)
        return results

    def compare_fertility(
        self,
        query_variants: Mapping[str, str],
        baseline_language: str = "en",
    ) -> dict[str, float]:
        """Compute relative subword fertility relative to the baseline language.

        Relative Fertility = Fertility(L) / Fertility(L_base)

        A relative fertility > 1.0 means the tokenizer fragments words in language L
        more aggressively than in the baseline language.
        """
        results = self.inspect_multilingual_query_group(query_variants)
        if baseline_language not in results:
            raise ValueError(f"Baseline language '{baseline_language}' not in query variants.")

        base_fertility = results[baseline_language].subword_fertility_ratio
        if base_fertility == 0.0:
            base_fertility = 1.0

        return {
            lang: round(res.subword_fertility_ratio / base_fertility, 2)
            for lang, res in results.items()
        }

    @staticmethod
    def get_methodological_limitations() -> dict[str, str]:
        """Return formal methodological limitations of tokenizer inspection.

        Scientific Caution:
            Token counts and subword fertility ratios must never be treated
            as a complete measure of linguistic understanding.
        """
        return {
            "construct_boundary": (
                "Subword fertility measures vocabulary fragmentation, not semantic comprehension. "
                "A model with a low fertility ratio for a language may still produce poor vector "
                "representations if its transformer layers were undertrained on that language."
            ),
            "script_asymmetry": (
                "Languages using non-Latin scripts (e.g., Bengali, Devanagari, Arabic) often "
                "receive smaller vocabulary allocations in multilingual tokenizers (like XLM-RoBERTa "
                "or SentencePiece), inflating token counts relative to Latin-script English."
            ),
            "transliteration_ambiguity": (
                "Phonetically transliterated queries (e.g., Banglish 'bn-Latn') are rarely seen "
                "in standard training corpora, causing tokenizers to fragment words into character-level "
                "n-grams or misclassify them as European language roots."
            ),
        }
