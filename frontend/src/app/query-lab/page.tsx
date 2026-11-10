"use client";

import * as React from "react";
import {
  Search,
  Sparkles,
  Layers,
  ArrowRight,
  Clock,
  SlidersHorizontal,
  ChevronDown,
  ChevronUp,
  Tag,
  Hash,
  AlertCircle,
  HelpCircle,
  CheckCircle2,
  Terminal,
} from "lucide-react";
import { api } from "@/lib/api";
import {
  ModelInfo,
  SearchResultItem,
  SearchResponse,
  TokenizationInspectResponse,
  SUPPORTED_LANGUAGES,
  RetrievalMode,
} from "@/types";
import {
  cn,
  formatLatency,
  isRtlLanguage,
  getLanguageFontClass,
  getLanguageMeta,
} from "@/lib/utils";

interface QuerySuggestion {
  language: string;
  label: string;
  query: string;
  translation: string;
}

const SAMPLE_SUGGESTIONS: QuerySuggestion[] = [
  {
    language: "en",
    label: "English (Formal)",
    query: "climate change effects on agricultural productivity",
    translation: "Standard baseline query",
  },
  {
    language: "bn",
    label: "Bengali (বাংলা)",
    query: "কৃষি ক্ষেত্রে জলবায়ু পরিবর্তনের প্রভাব",
    translation: "Climate change effects in agriculture",
  },
  {
    language: "bn-Latn",
    label: "Banglish (Transliteration)",
    query: "krishi khetre jolobayu poribortoner probhab",
    translation: "Phonetic transliteration of Bengali query",
  },
  {
    language: "hi",
    label: "Hindi (हिन्दी)",
    query: "कृषि पर जलवायु परिवर्तन का प्रभाव",
    translation: "Impact of climate change on agriculture",
  },
  {
    language: "ar",
    label: "Arabic (العربية)",
    query: "تأثير تغير المناخ على الإنتاجية الزراعية",
    translation: "Impact of climate change on agricultural productivity",
  },
  {
    language: "en",
    label: "English (Tech)",
    query: "neural machine translation subword tokenization methods",
    translation: "Technical NLP domain query",
  },
  {
    language: "bn",
    label: "Bengali (Tech)",
    query: "নিউরাল মেশিন অনুবাদ সাবওয়ার্ড টোকেনাইজেশন পদ্ধতি",
    translation: "Neural machine translation subword tokenization",
  },
];

export default function QueryLabPage() {
  // Query controls
  const [queryText, setQueryText] = React.useState(
    "climate change effects on agricultural productivity"
  );
  const [selectedModel, setSelectedModel] = React.useState<string>(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  );
  const [selectedLanguage, setSelectedLanguage] = React.useState<string>("en");
  const [topK, setTopK] = React.useState<number>(10);
  const [retrievalMode, setRetrievalMode] =
    React.useState<RetrievalMode>("EXACT");
  const [filterLanguage, setFilterLanguage] = React.useState<string>("");

  // Available models from backend
  const [models, setModels] = React.useState<ModelInfo[]>([]);

  // Search State
  const [searching, setSearching] = React.useState<boolean>(false);
  const [searchResponse, setSearchResponse] =
    React.useState<SearchResponse | null>(null);
  const [searchError, setSearchError] = React.useState<string | null>(null);

  // Tokenization State
  const [inspectTokenization, setInspectTokenization] =
    React.useState<boolean>(true);
  const [tokenData, setTokenData] =
    React.useState<TokenizationInspectResponse | null>(null);
  const [tokenLoading, setTokenLoading] = React.useState<boolean>(false);

  // Load models on mount
  React.useEffect(() => {
    async function loadModels() {
      try {
        const list = await api.getModels();
        if (list && list.length > 0) {
          setModels(list);
          // Set first model if current isn't valid
          setSelectedModel((prev) =>
            list.some((m) => m.model_id === prev) ? prev : list[0].model_id
          );
        }
      } catch (err) {
        console.error("Failed to load models list", err);
      }
    }
    loadModels();
  }, []);

  // Fetch token inspection when query or model changes (debounced)
  React.useEffect(() => {
    if (!inspectTokenization || !queryText.trim()) {
      setTokenData(null);
      return;
    }

    const timer = setTimeout(async () => {
      setTokenLoading(true);
      try {
        const res = await api.inspectTokenization({
          text: queryText,
          model_id: selectedModel,
          language_code: selectedLanguage,
        });
        setTokenData(res);
      } catch (e) {
        // Tokenizer inspection error
        console.warn("Token inspection error", e);
      } finally {
        setTokenLoading(false);
      }
    }, 400);

    return () => clearTimeout(timer);
  }, [queryText, selectedModel, selectedLanguage, inspectTokenization]);

  // Execute Search
  async function handleSearch(e?: React.FormEvent) {
    if (e) e.preventDefault();
    if (!queryText.trim()) return;

    setSearching(true);
    setSearchError(null);

    try {
      const response = await api.search({
        query_text: queryText.trim(),
        model_id: selectedModel,
        retrieval_mode: retrievalMode,
        top_k: topK,
        filter_language: filterLanguage ? filterLanguage : null,
      });
      setSearchResponse(response);
    } catch (err) {
      setSearchError(
        err instanceof Error
          ? err.message
          : "Search execution failed on the gateway."
      );
      setSearchResponse(null);
    } finally {
      setSearching(false);
    }
  }

  const isRtl = isRtlLanguage(selectedLanguage);
  const fontClass = getLanguageFontClass(selectedLanguage);

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      {/* Page Header */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-foreground">
          Multilingual Query Lab
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Interactive semantic vector search playground across multi-script
          corpora. Compare embedding models, test transliteration robustness,
          and inspect subword token fragmentation.
        </p>
      </div>

      {/* Main Two-Column Layout: Controls on Left/Top, Results on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Search Form & Tokenizer (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <form
            onSubmit={handleSearch}
            className="rounded-xl border border-border bg-card p-6 space-y-5 shadow-sm"
          >
            {/* Model Selector */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center justify-between">
                <span>Embedding Model</span>
                {models.length > 0 && (
                  <span className="text-[10px] text-primary lowercase font-mono">
                    {models.find((m) => m.model_id === selectedModel)
                      ?.embedding_dimension || 384}
                    d vector
                  </span>
                )}
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-3 py-2 text-sm rounded-lg border border-border bg-secondary/50 text-foreground focus:outline-none focus:ring-2 focus:ring-primary font-medium"
              >
                {models.length > 0 ? (
                  models.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.display_name} ({m.embedding_dimension}d)
                    </option>
                  ))
                ) : (
                  <>
                    <option value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2">
                      paraphrase-multilingual-MiniLM-L12-v2 (384d)
                    </option>
                    <option value="intfloat/multilingual-e5-small">
                      multilingual-e5-small (384d)
                    </option>
                  </>
                )}
              </select>
            </div>

            {/* Language Selector */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Query Language &amp; Script
              </label>
              <div className="grid grid-cols-3 sm:grid-cols-5 gap-1.5">
                {SUPPORTED_LANGUAGES.map((lang) => {
                  const isSelected = selectedLanguage === lang.code;
                  return (
                    <button
                      type="button"
                      key={lang.code}
                      onClick={() => setSelectedLanguage(lang.code)}
                      className={cn(
                        "px-2 py-2 text-xs rounded-lg border text-center transition flex flex-col items-center justify-center gap-0.5",
                        isSelected
                          ? "bg-primary text-primary-foreground border-primary font-semibold shadow-sm"
                          : "bg-secondary/40 text-muted-foreground border-border hover:text-foreground hover:bg-secondary"
                      )}
                    >
                      <span className="font-mono text-[10px] uppercase">
                        {lang.code}
                      </span>
                      <span className="text-[11px] font-medium truncate max-w-full">
                        {lang.nativeName.split(" ")[0]}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Query Input Box */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                <label htmlFor="query-textarea">Query Text</label>
                <span className="text-[11px] font-mono text-muted-foreground lowercase">
                  {queryText.length} chars
                </span>
              </div>
              <textarea
                id="query-textarea"
                rows={3}
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                dir={isRtl ? "rtl" : "ltr"}
                placeholder="Enter search query..."
                className={cn(
                  "w-full px-3.5 py-2.5 rounded-lg border border-border bg-secondary/30 text-foreground text-sm focus:outline-none focus:ring-2 focus:ring-primary transition resize-none",
                  fontClass,
                  isRtl ? "text-right" : "text-left"
                )}
              />
            </div>

            {/* Search Configuration Grid (Top-K & Mode) */}
            <div className="grid grid-cols-2 gap-3 pt-1">
              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">
                  Top-K Depth
                </label>
                <select
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full px-2.5 py-1.5 text-xs rounded-lg border border-border bg-secondary/50 text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                >
                  <option value={5}>Top 5</option>
                  <option value={10}>Top 10</option>
                  <option value={20}>Top 20</option>
                  <option value={50}>Top 50</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-xs font-medium text-muted-foreground">
                  Retrieval Mode
                </label>
                <select
                  value={retrievalMode}
                  onChange={(e) =>
                    setRetrievalMode(e.target.value as RetrievalMode)
                  }
                  className="w-full px-2.5 py-1.5 text-xs rounded-lg border border-border bg-secondary/50 text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                >
                  <option value="EXACT">EXACT (Brute-force)</option>
                  <option value="VECTOR_INDEX">VECTOR_INDEX (HNSW)</option>
                </select>
              </div>
            </div>

            {/* Optional Language Filter */}
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground flex items-center justify-between">
                <span>Corpus Language Filter (Optional)</span>
                <span className="text-[10px] text-muted-foreground">
                  Leave empty for cross-lingual
                </span>
              </label>
              <select
                value={filterLanguage}
                onChange={(e) => setFilterLanguage(e.target.value)}
                className="w-full px-2.5 py-1.5 text-xs rounded-lg border border-border bg-secondary/50 text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
              >
                <option value="">All Languages (Cross-Lingual Retrieval)</option>
                <option value="en">English (en)</option>
                <option value="bn">Bengali (bn)</option>
                <option value="bn-Latn">Banglish (bn-Latn)</option>
                <option value="hi">Hindi (hi)</option>
                <option value="ar">Arabic (ar)</option>
              </select>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={searching || !queryText.trim()}
              className="w-full py-2.5 px-4 rounded-lg bg-primary text-primary-foreground font-semibold text-sm hover:bg-primary/90 disabled:opacity-50 transition shadow-sm shadow-primary/20 flex items-center justify-center gap-2"
            >
              <Search className={`h-4 w-4 ${searching ? "animate-spin" : ""}`} />
              {searching ? "Encoding & Searching..." : "Execute Semantic Search"}
            </button>
          </form>

          {/* Selectable Query Suggestions */}
          <div className="rounded-xl border border-border bg-card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                <Sparkles className="h-3.5 w-3.5 text-primary" />
                Benchmark Query Suggestions
              </span>
              <span className="text-[11px] text-muted-foreground">
                Click to load (no auto-run)
              </span>
            </div>

            <div className="space-y-2">
              {SAMPLE_SUGGESTIONS.map((s, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => {
                    setQueryText(s.query);
                    setSelectedLanguage(s.language);
                  }}
                  className="w-full text-left p-2.5 rounded-lg border border-border bg-secondary/20 hover:bg-secondary/60 hover:border-primary/40 transition group"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-semibold text-primary">
                      {s.label}
                    </span>
                    <span className="text-[10px] text-muted-foreground">
                      {s.translation}
                    </span>
                  </div>
                  <p
                    className={cn(
                      "text-xs text-foreground mt-1 truncate",
                      getLanguageFontClass(s.language)
                    )}
                    dir={isRtlLanguage(s.language) ? "rtl" : "ltr"}
                  >
                    {s.query}
                  </p>
                </button>
              ))}
            </div>
          </div>

          {/* Live Tokenizer Inspector Card */}
          <div className="rounded-xl border border-border bg-card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                <Terminal className="h-3.5 w-3.5 text-violet-400" />
                Tokenizer Inspector (Subword Fertility)
              </span>
              <button
                type="button"
                onClick={() => setInspectTokenization(!inspectTokenization)}
                className="text-[11px] text-primary hover:underline"
              >
                {inspectTokenization ? "Hide" : "Show"}
              </button>
            </div>

            {inspectTokenization && (
              <div className="space-y-3">
                {tokenLoading ? (
                  <p className="text-xs text-muted-foreground">
                    Tokenizing surface string...
                  </p>
                ) : tokenData ? (
                  <div className="space-y-3">
                    {/* Metrics row */}
                    <div className="grid grid-cols-3 gap-2 p-2.5 rounded-lg bg-secondary/40 border border-border text-center">
                      <div>
                        <div className="text-[10px] text-muted-foreground">
                          Word Count
                        </div>
                        <div className="text-sm font-semibold font-mono text-foreground">
                          {tokenData.word_count}
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-muted-foreground">
                          Subwords
                        </div>
                        <div className="text-sm font-semibold font-mono text-foreground">
                          {tokenData.text_token_count}
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-muted-foreground">
                          Fertility
                        </div>
                        <div className="text-sm font-semibold font-mono text-primary">
                          {tokenData.subword_fertility_ratio.toFixed(2)}x
                        </div>
                      </div>
                    </div>

                    {/* Subword token fragments */}
                    <div>
                      <div className="text-[11px] text-muted-foreground mb-1.5">
                        Extracted Subwords:
                      </div>
                      <div className="flex flex-wrap gap-1 max-h-32 overflow-y-auto p-1.5 rounded-lg bg-secondary/30 border border-border">
                        {tokenData.tokens.map((tok, i) => (
                          <span
                            key={i}
                            className={cn(
                              "px-1.5 py-0.5 rounded text-[11px] font-mono border",
                              tok.startsWith("##") || tok.startsWith(" ")
                                ? "bg-primary/10 text-primary border-primary/20"
                                : tok === "<s>" || tok === "</s>" || tok === "[CLS]" || tok === "[SEP]"
                                ? "bg-secondary text-muted-foreground border-border"
                                : "bg-card text-foreground border-border"
                            )}
                          >
                            {tok}
                          </span>
                        ))}
                      </div>
                    </div>

                    <p className="text-[11px] text-muted-foreground leading-relaxed italic">
                      {tokenData.limitations_note}
                    </p>
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    Type a query to preview subword segmentation.
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Search Results (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          {/* Results Summary Bar */}
          <div className="flex items-center justify-between p-4 rounded-xl border border-border bg-card">
            <div>
              <h3 className="text-sm font-semibold text-foreground">
                Candidate Document Results
              </h3>
              <p className="text-xs text-muted-foreground">
                {searchResponse
                  ? `Retrieved ${searchResponse.count} matches in ${formatLatency(
                      searchResponse.latency_ms
                    )}`
                  : "Execute a query to retrieve ranked candidates."}
              </p>
            </div>

            {searchResponse && (
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 text-[11px] font-mono rounded bg-secondary text-muted-foreground border border-border">
                  mode: {searchResponse.retrieval_mode}
                </span>
                <span className="px-2 py-0.5 text-[11px] font-mono rounded bg-primary/10 text-primary border border-primary/20 font-semibold">
                  {formatLatency(searchResponse.latency_ms)}
                </span>
              </div>
            )}
          </div>

          {/* Search Error State */}
          {searchError && (
            <div className="p-4 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-400 text-xs flex items-start gap-3">
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Search Execution Error</p>
                <p className="mt-0.5">{searchError}</p>
              </div>
            </div>
          )}

          {/* Results List */}
          {searchResponse && searchResponse.results.length > 0 ? (
            <div className="space-y-3">
              {searchResponse.results.map((item: SearchResultItem) => {
                const langMeta = getLanguageMeta(item.language_code);
                const isRtlDoc = isRtlLanguage(item.language_code);
                const fontDoc = getLanguageFontClass(item.language_code);

                return (
                  <div
                    key={item.document_id}
                    className="rounded-xl border border-border bg-card p-5 space-y-3 hover:border-primary/40 transition-colors"
                  >
                    {/* Header Row: Rank, External ID, Language Badge, Similarity Score */}
                    <div className="flex items-center justify-between gap-2 flex-wrap">
                      <div className="flex items-center gap-2">
                        <span className="inline-flex items-center justify-center h-6 w-6 rounded-md bg-secondary text-foreground text-xs font-bold font-mono">
                          #{item.rank}
                        </span>
                        <span className="text-xs font-mono text-muted-foreground">
                          {item.doc_external_id}
                        </span>
                        <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-primary/10 text-primary border border-primary/20">
                          {item.language_code} ({langMeta.label})
                        </span>
                      </div>

                      {/* Cosine Similarity Score Badge */}
                      <div className="flex items-center gap-1.5">
                        <span className="text-[11px] text-muted-foreground">
                          Cosine Sim:
                        </span>
                        <span className="px-2 py-0.5 text-xs font-bold font-mono rounded bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                          {item.score.toFixed(4)}
                        </span>
                      </div>
                    </div>

                    {/* Document Title if present */}
                    {item.title && (
                      <h4 className="text-sm font-semibold text-foreground">
                        {item.title}
                      </h4>
                    )}

                    {/* Document Content */}
                    <p
                      className={cn(
                        "text-sm text-foreground/90 leading-relaxed",
                        fontDoc,
                        isRtlDoc ? "text-right" : "text-left"
                      )}
                      dir={isRtlDoc ? "rtl" : "ltr"}
                    >
                      {item.content}
                    </p>

                    {/* Metadata footer */}
                    <div className="pt-2 border-t border-border/60 flex items-center justify-between text-[11px] text-muted-foreground">
                      <span className="font-mono truncate">
                        UUID: {item.document_id}
                      </span>
                      <span>Script: {item.script_code}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : searchResponse && searchResponse.results.length === 0 ? (
            <div className="rounded-xl border border-dashed border-border bg-card/40 p-12 text-center space-y-3">
              <Search className="h-8 w-8 text-muted-foreground mx-auto stroke-1" />
              <p className="text-sm font-semibold text-foreground">
                No matching documents retrieved
              </p>
              <p className="text-xs text-muted-foreground max-w-sm mx-auto">
                No indexed documents matched the search criteria. If your
                corpus is empty, please ingest or index documents via Dataset
                Studio.
              </p>
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-border bg-card/40 p-12 text-center space-y-3">
              <SlidersHorizontal className="h-8 w-8 text-muted-foreground mx-auto stroke-1" />
              <p className="text-sm font-semibold text-foreground">
                Search Results Display
              </p>
              <p className="text-xs text-muted-foreground max-w-sm mx-auto">
                Select an embedding model and query above, then click &ldquo;Execute
                Semantic Search&rdquo; to review real-time vector matches and
                fertility metrics.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
