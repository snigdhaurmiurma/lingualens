/**
 * LinguaLens TypeScript Type Definitions
 * Directly aligned with FastAPI Pydantic schemas in backend/app/schemas/
 */

export type LanguageCode = "en" | "bn" | "bn-Latn" | "hi" | "ar" | string;
export type ScriptCode = "Latn" | "Beng" | "Deva" | "Arab" | string;
export type RetrievalMode = "EXACT" | "VECTOR_INDEX";

export interface LanguageMeta {
  code: LanguageCode;
  label: string;
  nativeName: string;
  script: ScriptCode;
  isRtl?: boolean;
}

export const SUPPORTED_LANGUAGES: LanguageMeta[] = [
  { code: "en", label: "English", nativeName: "English", script: "Latn", isRtl: false },
  { code: "bn", label: "Bengali", nativeName: "বাংলা", script: "Beng", isRtl: false },
  { code: "bn-Latn", label: "Banglish", nativeName: "Banglish (Transliteration)", script: "Latn", isRtl: false },
  { code: "hi", label: "Hindi", nativeName: "हिन्दी", script: "Deva", isRtl: false },
  { code: "ar", label: "Arabic", nativeName: "العربية", script: "Arab", isRtl: true },
];

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
  database: "connected" | "disconnected" | string;
}

export interface SystemInfoResponse {
  python_version: string;
  torch_version: string;
  cpu_count: number;
  torch_cpu_threads: number;
  device_default: string;
  memory_available_mb?: number | null;
  registered_models: string[];
}

export interface ModelInfo {
  model_id: string;
  display_name: string;
  provider: string;
  parameter_count: number;
  embedding_dimension: number;
  max_sequence_length: number;
  license: string;
  query_prefix: string;
  document_prefix: string;
  supported_languages: string[];
  requires_gpu: boolean;
  is_loaded: boolean;
}

export interface Dataset {
  id: string;
  name: string;
  slug: string;
  version: string;
  description?: string | null;
  dataset_type: "VALIDATED_BENCHMARK" | "DEMONSTRATION" | string;
  license: string;
  supported_languages: string[];
  content_hash_sha256: string;
  is_locked: boolean;
  document_count: number;
  query_count: number;
  query_group_count: number;
  created_at: string;
}

export interface PaginatedResponse<T> {
  total: number;
  limit: number;
  offset: number;
  items: T[];
}

export interface SearchResultItem {
  document_id: string;
  doc_external_id: string;
  score: number;
  rank: number;
  language_code: string;
  script_code: string;
  title?: string | null;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface SearchResponse {
  query_text: string;
  model_id: string;
  retrieval_mode: RetrievalMode;
  top_k: number;
  count: number;
  latency_ms: float_number;
  results: SearchResultItem[];
}

type float_number = number;

export interface SearchRequest {
  query_text: string;
  model_id?: string;
  retrieval_mode?: RetrievalMode;
  top_k?: number;
  filter_language?: string | null;
  filter_dataset_id?: string | null;
}

export interface TokenizationInspectRequest {
  text: string;
  model_id?: string;
  language_code?: string;
}

export interface TokenizationInspectResponse {
  model_id: string;
  language_code: string;
  text: string;
  tokens: string[];
  token_count: number;
  text_tokens: string[];
  text_token_count: number;
  word_count: number;
  subword_fertility_ratio: number;
  has_unknown_tokens: boolean;
  unknown_token_count: number;
  limitations_note: string;
}

export interface ExperimentRun {
  id: string;
  experiment_id: string;
  model_id: string;
  retrieval_mode: RetrievalMode;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | string;
  progress_percent: number;
  error_message?: string | null;
  device_used: string;
  hardware_metadata?: Record<string, unknown> | null;
  total_latency_seconds?: number | null;
  created_at: string;
}

export interface Experiment {
  id: string;
  name: string;
  description?: string | null;
  dataset_id: string;
  baseline_language_code: string;
  cutoff_k_values: number[];
  similarity_metric: string;
  created_at: string;
  runs: ExperimentRun[];
}

export interface MetricResultItem {
  language_code: string;
  script_code: string;
  metric_name: string;
  metric_value: number;
  ci_lower_95?: number | null;
  ci_upper_95?: number | null;
  p_value_vs_baseline?: number | null;
  is_statistically_significant: boolean;
}

export interface FairnessDisparityItem {
  language_code: string;
  baseline_language: string;
  absolute_performance_gap: number;
  gap_ci_lower_95?: number | null;
  gap_ci_upper_95?: number | null;
  relative_disparity_ratio: number;
  p_value?: number | null;
  is_statistically_significant: boolean;
}

export interface ExperimentEvaluationSummary {
  experiment_id: string;
  experiment_name: string;
  dataset_id: string;
  baseline_language: string;
  runs: ExperimentRun[];
  metrics: MetricResultItem[];
  fairness_disparities: FairnessDisparityItem[];
  transliteration_penalties: Record<string, number>;
}

export interface StreamEvent {
  event: string;
  run_id: string;
  timestamp: string;
  percent: number;
  stage: string;
  detail: string;
  payload?: Record<string, unknown>;
}
