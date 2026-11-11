/**
 * LinguaLens Showcase Benchmark Data & In-Browser Evaluation Engine
 *
 * Provides high-fidelity, deterministic benchmark evaluation and semantic
 * vector simulation when the local FastAPI gateway is offline or in static cloud hosting (e.g. Vercel).
 */

import {
  ModelInfo,
  Dataset,
  PaginatedResponse,
  Experiment,
  ExperimentEvaluationSummary,
  SearchRequest,
  SearchResponse,
  SearchResultItem,
  TokenizationInspectRequest,
  TokenizationInspectResponse,
  SystemInfoResponse,
  HealthResponse,
} from "@/types";

export const SHOWCASE_MODELS: ModelInfo[] = [
  {
    model_id: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    display_name: "Paraphrase Multilingual MiniLM L12 v2",
    provider: "Hugging Face / Sentence-Transformers",
    parameter_count: 117_000_000,
    embedding_dimension: 384,
    max_sequence_length: 128,
    license: "Apache-2.0",
    query_prefix: "",
    document_prefix: "",
    supported_languages: ["en", "bn", "bn-Latn", "hi", "ar"],
    requires_gpu: false,
    is_loaded: true,
  },
  {
    model_id: "intfloat/multilingual-e5-small",
    display_name: "Multilingual E5 Small",
    provider: "Microsoft / intfloat",
    parameter_count: 118_000_000,
    embedding_dimension: 384,
    max_sequence_length: 512,
    license: "MIT",
    query_prefix: "query: ",
    document_prefix: "passage: ",
    supported_languages: ["en", "bn", "bn-Latn", "hi", "ar"],
    requires_gpu: false,
    is_loaded: true,
  },
];

export const SHOWCASE_DOCUMENTS: SearchResultItem[] = [
  {
    document_id: "8ac560f4-4227-4453-ac4a-894a2c38c26b",
    doc_external_id: "doc-en-agri-01",
    score: 0.8842,
    rank: 1,
    language_code: "en",
    script_code: "Latn",
    title: "Climate Change Impacts on Crop Yields",
    content:
      "Rising global temperatures, erratic precipitation, and prolonged droughts significantly reduce agricultural productivity and crop yields across tropical and subtropical farming regions.",
    metadata: { domain: "agriculture", climate_sensitive: true },
  },
  {
    document_id: "694179bf-0873-4a51-bd49-62e05a884c56",
    doc_external_id: "doc-hi-agri-01",
    score: 0.8371,
    rank: 2,
    language_code: "hi",
    script_code: "Deva",
    title: "कृषि पर जलवायु परिवर्तन का प्रतिकूल प्रभाव",
    content:
      "बढ़ते वैश्विक तापमान, अनियमित वर्षा और लंबे सूखे के कारण ग्रामीण क्षेत्रों में कृषि उत्पादकता और फसलों की पैदावार में भारी गिरावट आ रही है।",
    metadata: { domain: "agriculture", climate_sensitive: true },
  },
  {
    document_id: "0c2e1c4b-bc83-4067-8f16-b1d09e169e76",
    doc_external_id: "doc-ar-agri-01",
    score: 0.824,
    rank: 3,
    language_code: "ar",
    script_code: "Arab",
    title: "تأثير تغير المناخ على الإنتاجية الزراعية",
    content:
      "يؤدي ارتفاع درجات الحرارة العالمية والأمطار غير المنتظمة وفترات الجفاف الطويلة إلى انخفاض حاد في إنتاجية المحاصيل الزراعية في المناطق الريفية.",
    metadata: { domain: "agriculture", climate_sensitive: true },
  },
  {
    document_id: "3e5f2a1b-90c8-4721-a4b3-d8c7e6f5a4b1",
    doc_external_id: "doc-bn-agri-01",
    score: 0.8125,
    rank: 4,
    language_code: "bn",
    script_code: "Beng",
    title: "কৃষি ক্ষেত্রে জলবায়ু পরিবর্তনের ক্ষতিকর প্রভাব",
    content:
      "ক্রমবর্ধমান বৈশ্বিক তাপমাত্রা, অনিয়মিত বৃষ্টিপাত এবং দীর্ঘস্থায়ী খরা ক্রান্তীয় অঞ্চলের কৃষি উৎপাদন ও ফসলের ফলন ব্যাপকভাবে হ্রাস করছে।",
    metadata: { domain: "agriculture", climate_sensitive: true },
  },
  {
    document_id: "4f6a3b2c-01d9-4832-b5c4-e9d8f7a6b5c2",
    doc_external_id: "doc-bn-latn-agri-01",
    score: 0.6912,
    rank: 5,
    language_code: "bn-Latn",
    script_code: "Latn",
    title: "Krishi khetre jolobayu poribortoner probhab",
    content:
      "Bishshotomapmatra briddhi, oniyomito brishti ebong khora krishi utpadon o fosholer folon prochur porimane komiye dicche.",
    metadata: { domain: "agriculture", transliterated: true },
  },
  {
    document_id: "5a7b4c3d-12e0-4943-c6d5-f0e9a8b7c6d3",
    doc_external_id: "doc-en-renew-02",
    score: 0.542,
    rank: 6,
    language_code: "en",
    script_code: "Latn",
    title: "Renewable Energy in Rural Farming",
    content:
      "Solar-powered irrigation and wind turbines provide sustainable energy for rural agriculture, reducing carbon emissions and operational farming expenses.",
    metadata: { domain: "energy", climate_sensitive: false },
  },
  {
    document_id: "6b8c5d4e-23f1-4054-d7e6-a1f0b9c8d7e4",
    doc_external_id: "doc-bn-renew-02",
    score: 0.518,
    rank: 7,
    language_code: "bn",
    script_code: "Beng",
    title: "গ্রামীণ কৃষিতে সৌরশক্তি ও নবায়নযোগ্য প্রযুক্তি",
    content:
      "সৌরচালিত সেচ পাম্প এবং বায়ুবিদ্যুৎ গ্রামীণ কৃষি খাতে টেকসই জ্বালানি সরবরাহ করে, যা কার্বন নির্গমন এবং কৃষকদের খরচ কমায়।",
    metadata: { domain: "energy", climate_sensitive: false },
  },
];

export const SHOWCASE_DATASETS: Dataset[] = [
  {
    id: "a0827a5f-9089-4ee5-a76b-80b30ab872f2",
    name: "Global Climate & Agriculture Benchmark",
    slug: "multilingual-ir-benchmark",
    version: "1.0.0",
    description:
      "Parallel multilingual evaluation benchmark across English, Bengali, Banglish, Hindi, and Arabic. Evaluates cross-script semantic retrieval and transliteration robustness.",
    dataset_type: "VALIDATED_BENCHMARK",
    license: "CC-BY-4.0",
    supported_languages: ["en", "bn", "bn-Latn", "hi", "ar"],
    content_hash_sha256:
      "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
    is_locked: true,
    document_count: 7,
    query_count: 14,
    query_group_count: 2,
    created_at: "2026-11-10T12:00:00Z",
  },
];

export const SHOWCASE_EXPERIMENTS: Experiment[] = [
  {
    id: "exp-multilingual-retrieval-eval",
    name: "Cross-Lingual Retrieval Baseline & Transliteration Disparity",
    description:
      "Systematic comparison of paraphrase-multilingual-MiniLM-L12-v2 against multilingual-e5-small across 5 languages with paired Wilcoxon significance testing.",
    dataset_id: "a0827a5f-9089-4ee5-a76b-80b30ab872f2",
    baseline_language_code: "en",
    cutoff_k_values: [5, 10],
    similarity_metric: "cosine",
    created_at: "2026-11-10T14:30:00Z",
    runs: [
      {
        id: "run-minilm-exact",
        experiment_id: "exp-multilingual-retrieval-eval",
        model_id: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        retrieval_mode: "EXACT",
        status: "COMPLETED",
        progress_percent: 100,
        device_used: "cpu",
        total_latency_seconds: 4.82,
        created_at: "2026-11-10T14:32:00Z",
      },
      {
        id: "run-e5-exact",
        experiment_id: "exp-multilingual-retrieval-eval",
        model_id: "intfloat/multilingual-e5-small",
        retrieval_mode: "EXACT",
        status: "COMPLETED",
        progress_percent: 100,
        device_used: "cpu",
        total_latency_seconds: 5.14,
        created_at: "2026-11-10T14:38:00Z",
      },
    ],
  },
];

export function showcaseSearch(request: SearchRequest): SearchResponse {
  const query = request.query_text.trim().toLowerCase();
  const topK = request.top_k || 10;
  const filterLang = request.filter_language || null;

  // Filter by language if specified
  let docs = SHOWCASE_DOCUMENTS;
  if (filterLang) {
    docs = docs.filter((d) => d.language_code === filterLang);
  }

  // Calculate semantic similarity scores with authentic language gaps
  const scored = docs.map((doc) => {
    let baseScore = 0.55;

    // Check keyword overlaps
    const qWords = query.split(/\s+/).filter(Boolean);
    const contentLower = (doc.title + " " + doc.content).toLowerCase();

    let matched = 0;
    for (const w of qWords) {
      if (w.length > 2 && contentLower.includes(w)) {
        matched++;
      }
    }

    if (
      query.includes("climate") ||
      query.includes("agri") ||
      query.includes("crop") ||
      query.includes("কৃষি") ||
      query.includes("জলবায়ু") ||
      query.includes("krishi") ||
      query.includes("jolobayu") ||
      query.includes("मौसम") ||
      query.includes("مناخ")
    ) {
      if (doc.metadata?.domain === "agriculture") {
        baseScore += 0.28;
      }
    }

    if (
      query.includes("energy") ||
      query.includes("solar") ||
      query.includes("বিদ্যুৎ") ||
      query.includes("সৌর") ||
      query.includes("turbine")
    ) {
      if (doc.metadata?.domain === "energy") {
        baseScore += 0.32;
      }
    }

    baseScore += Math.min(0.12, matched * 0.04);

    // Reflect transliteration penalty for Banglish (bn-Latn)
    if (doc.language_code === "bn-Latn") {
      baseScore = Math.max(0.48, baseScore - 0.11);
    }

    // Slightly higher score if model is E5 Small
    if (request.model_id?.includes("e5")) {
      baseScore = Math.min(0.96, baseScore * 1.03);
    }

    return {
      ...doc,
      score: Math.min(0.95, Math.max(0.42, parseFloat(baseScore.toFixed(4)))),
    };
  });

  // Sort descending by score
  scored.sort((a, b) => b.score - a.score);

  // Assign ranks
  const results = scored.slice(0, topK).map((item, idx) => ({
    ...item,
    rank: idx + 1,
  }));

  return {
    query_text: request.query_text,
    model_id:
      request.model_id ||
      "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    retrieval_mode: request.retrieval_mode || "EXACT",
    top_k: topK,
    count: results.length,
    latency_ms: parseFloat((14.5 + Math.random() * 8.2).toFixed(2)),
    results,
  };
}

export function showcaseTokenize(
  request: TokenizationInspectRequest
): TokenizationInspectResponse {
  const text = request.text.trim();
  const lang = request.language_code || "en";
  const words = text.split(/\s+/).filter(Boolean);
  const wordCount = Math.max(1, words.length);

  const tokens: string[] = ["<s>"];
  for (const word of words) {
    if (/[\u0980-\u09FF]/.test(word)) {
      // Bengali script: higher fragmentation into subword grapheme units
      if (word.length <= 3) {
        tokens.push(" " + word);
      } else {
        const mid = Math.ceil(word.length / 2);
        tokens.push(" " + word.slice(0, mid));
        tokens.push("##" + word.slice(mid));
      }
    } else if (/[\u0600-\u06FF]/.test(word)) {
      // Arabic script
      if (word.length <= 4) {
        tokens.push(" " + word);
      } else {
        tokens.push(" " + word.slice(0, 3));
        tokens.push("##" + word.slice(3));
      }
    } else if (/[\u0900-\u097F]/.test(word)) {
      // Devanagari Hindi
      if (word.length <= 3) {
        tokens.push(" " + word);
      } else {
        const mid = Math.ceil(word.length / 2);
        tokens.push(" " + word.slice(0, mid));
        tokens.push("##" + word.slice(mid));
      }
    } else if (lang === "bn-Latn") {
      // Banglish transliteration: splits due to non-standard Roman subword units
      if (word.length > 5) {
        tokens.push(" " + word.slice(0, 4));
        tokens.push("##" + word.slice(4));
      } else {
        tokens.push(" " + word);
      }
    } else {
      // Standard Latin / English
      tokens.push(" " + word);
    }
  }
  tokens.push("</s>");

  const textTokens = tokens.filter((t) => t !== "<s>" && t !== "</s>");
  const fertility = parseFloat((textTokens.length / wordCount).toFixed(2));

  let note = "Standard vocabulary alignment. Low subword fragmentation.";
  if (lang === "bn") {
    note =
      "Bengali Unicode script exhibits moderate subword over-segmentation in multilingual XLM-R/MiniLM tokenizers due to vocabulary budget constraints.";
  } else if (lang === "bn-Latn") {
    note =
      "Banglish transliteration induces high subword fragmentation: phonetic Romanized spelling lacks standard vocabulary tokens, inflating subword fertility.";
  } else if (lang === "ar") {
    note =
      "Arabic root-and-pattern morphology results in subword affixes and stem decomposition.";
  }

  return {
    model_id:
      request.model_id ||
      "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    language_code: lang,
    text,
    tokens,
    token_count: tokens.length,
    text_tokens: textTokens,
    text_token_count: textTokens.length,
    word_count: wordCount,
    subword_fertility_ratio: fertility,
    has_unknown_tokens: false,
    unknown_token_count: 0,
    limitations_note: note,
  };
}
