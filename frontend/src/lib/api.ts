/**
 * Typed API Client for LinguaLens Backend Gateway
 *
 * Implements dual-mode connectivity:
 * 1. High-speed direct REST & SSE connection to local FastAPI gateway (http://localhost:8000).
 * 2. Automatic, deterministic in-browser Showcase Benchmark fallback when deployed on
 *    static cloud hosting (Vercel) or when external visitors browse without a local Python server.
 */

import {
  HealthResponse,
  SystemInfoResponse,
  ModelInfo,
  Dataset,
  PaginatedResponse,
  Experiment,
  ExperimentRun,
  ExperimentEvaluationSummary,
  SearchRequest,
  SearchResponse,
  TokenizationInspectRequest,
  TokenizationInspectResponse,
  StreamEvent,
} from "@/types";

import {
  SHOWCASE_MODELS,
  SHOWCASE_DATASETS,
  SHOWCASE_EXPERIMENTS,
  showcaseSearch,
  showcaseTokenize,
} from "./showcase-data";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") ||
  "http://localhost:8000/api/v1";

export class ApiError extends Error {
  status: number;
  data: unknown;

  constructor(message: string, status: number, data?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }
}

async function fetchJson<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint.startsWith("/") ? "" : "/"}${endpoint}`;
  const headers = {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...options.headers,
  };

  try {
    const res = await fetch(url, {
      ...options,
      headers,
    });

    if (!res.ok) {
      let errorBody: unknown;
      try {
        errorBody = await res.json();
      } catch {
        errorBody = await res.text();
      }
      throw new ApiError(
        `API request failed with status ${res.status}`,
        res.status,
        errorBody
      );
    }

    return (await res.json()) as T;
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      err instanceof Error ? err.message : "Network error connecting to LinguaLens gateway",
      0,
      err
    );
  }
}

function shouldRethrow(err: unknown): boolean {
  return err instanceof ApiError && err.status !== 0;
}

export const api = {
  // System & Health
  async getHealth(): Promise<HealthResponse> {
    try {
      return await fetchJson<HealthResponse>("/health");
    } catch (err) {
      if (shouldRethrow(err)) throw err;
      return {
        status: "healthy",
        version: "0.1.0-showcase",
        timestamp: new Date().toISOString(),
        database: "showcase-benchmark",
      };
    }
  },

  async getSystemInfo(): Promise<SystemInfoResponse> {
    try {
      return await fetchJson<SystemInfoResponse>("/system/info");
    } catch (err) {
      if (shouldRethrow(err)) throw err;
      return {
        python_version: "3.11.9 (Showcase Environment)",
        torch_version: "2.2.0 (CPU Runtime)",
        cpu_count: 8,
        torch_cpu_threads: 4,
        device_default: "cpu",
        memory_available_mb: 8192,
        registered_models: SHOWCASE_MODELS.map((m) => m.model_id),
      };
    }
  },

  // Embedding Models
  async getModels(): Promise<ModelInfo[]> {
    try {
      const models = await fetchJson<ModelInfo[]>("/models");
      if (models && models.length > 0) return models;
      return SHOWCASE_MODELS;
    } catch (err) {
      if (shouldRethrow(err)) throw err;
      return SHOWCASE_MODELS;
    }
  },

  // Datasets
  async getDatasets(
    limit = 50,
    offset = 0
  ): Promise<PaginatedResponse<Dataset>> {
    try {
      return await fetchJson<PaginatedResponse<Dataset>>(
        `/datasets?limit=${limit}&offset=${offset}`
      );
    } catch (err) {
      if (shouldRethrow(err)) throw err;
      return {
        total: SHOWCASE_DATASETS.length,
        limit,
        offset,
        items: SHOWCASE_DATASETS.slice(offset, offset + limit),
      };
    }
  },

  async getDataset(id: string): Promise<Dataset> {
    try {
      return await fetchJson<Dataset>(`/datasets/${id}`);
    } catch (err) {
      if (shouldRethrow(err)) throw err;
      const match = SHOWCASE_DATASETS.find((d) => d.id === id || d.slug === id);
      return match || SHOWCASE_DATASETS[0];
    }
  },

  // Experiments
  async getExperiments(
    limit = 50,
    offset = 0
  ): Promise<PaginatedResponse<Experiment>> {
    try {
      return await fetchJson<PaginatedResponse<Experiment>>(
        `/experiments?limit=${limit}&offset=${offset}`
      );
    } catch (err) {
      if (shouldRethrow(err)) throw err;
      return {
        total: SHOWCASE_EXPERIMENTS.length,
        limit,
        offset,
        items: SHOWCASE_EXPERIMENTS.slice(offset, offset + limit),
      };
    }
  },

  async getExperiment(id: string): Promise<Experiment> {
    try {
      return await fetchJson<Experiment>(`/experiments/${id}`);
    } catch (err) {
      if (shouldRethrow(err)) throw err;
      const match = SHOWCASE_EXPERIMENTS.find((e) => e.id === id);
      return match || SHOWCASE_EXPERIMENTS[0];
    }
  },

  async getExperimentSummary(
    id: string
  ): Promise<ExperimentEvaluationSummary> {
    try {
      return await fetchJson<ExperimentEvaluationSummary>(
        `/experiments/${id}/summary`
      );
    } catch (err) {
      if (shouldRethrow(err)) throw err;
      return {
        experiment_id: id,
        experiment_name: "Cross-Lingual Retrieval Baseline & Transliteration Disparity",
        dataset_id: "a0827a5f-9089-4ee5-a76b-80b30ab872f2",
        baseline_language: "en",
        runs: SHOWCASE_EXPERIMENTS[0].runs,
        metrics: [
          {
            language_code: "en",
            script_code: "Latn",
            metric_name: "Recall@10",
            metric_value: 0.884,
            ci_lower_95: 0.82,
            ci_upper_95: 0.94,
            p_value_vs_baseline: 1.0,
            is_statistically_significant: false,
          },
          {
            language_code: "hi",
            script_code: "Deva",
            metric_name: "Recall@10",
            metric_value: 0.837,
            ci_lower_95: 0.77,
            ci_upper_95: 0.9,
            p_value_vs_baseline: 0.042,
            is_statistically_significant: true,
          },
          {
            language_code: "ar",
            script_code: "Arab",
            metric_name: "Recall@10",
            metric_value: 0.824,
            ci_lower_95: 0.75,
            ci_upper_95: 0.89,
            p_value_vs_baseline: 0.038,
            is_statistically_significant: true,
          },
          {
            language_code: "bn",
            script_code: "Beng",
            metric_name: "Recall@10",
            metric_value: 0.812,
            ci_lower_95: 0.74,
            ci_upper_95: 0.88,
            p_value_vs_baseline: 0.031,
            is_statistically_significant: true,
          },
          {
            language_code: "bn-Latn",
            script_code: "Latn",
            metric_name: "Recall@10",
            metric_value: 0.691,
            ci_lower_95: 0.61,
            ci_upper_95: 0.77,
            p_value_vs_baseline: 0.0004,
            is_statistically_significant: true,
          },
        ],
        fairness_disparities: [
          {
            language_code: "bn",
            baseline_language: "en",
            absolute_performance_gap: 0.072,
            gap_ci_lower_95: 0.02,
            gap_ci_upper_95: 0.13,
            relative_disparity_ratio: 0.081,
            p_value: 0.031,
            is_statistically_significant: true,
          },
          {
            language_code: "bn-Latn",
            baseline_language: "en",
            absolute_performance_gap: 0.193,
            gap_ci_lower_95: 0.12,
            gap_ci_upper_95: 0.27,
            relative_disparity_ratio: 0.218,
            p_value: 0.0004,
            is_statistically_significant: true,
          },
        ],
        transliteration_penalties: {
          "bn vs bn-Latn": 0.121,
        },
      };
    }
  },

  async runExperiment(
    experimentId: string,
    params: {
      model_id?: string;
      retrieval_mode?: string;
      random_seed?: number;
    } = {}
  ): Promise<ExperimentRun> {
    try {
      return await fetchJson<ExperimentRun>(`/experiments/${experimentId}/run`, {
        method: "POST",
        body: JSON.stringify(params),
      });
    } catch (err) {
      if (shouldRethrow(err)) throw err;
      return {
        id: `run-${Date.now()}`,
        experiment_id: experimentId,
        model_id: params.model_id || SHOWCASE_MODELS[0].model_id,
        retrieval_mode: (params.retrieval_mode as any) || "EXACT",
        status: "COMPLETED",
        progress_percent: 100,
        device_used: "cpu",
        total_latency_seconds: 3.42,
        created_at: new Date().toISOString(),
      };
    }
  },

  // Semantic Search
  async search(request: SearchRequest): Promise<SearchResponse> {
    try {
      return await fetchJson<SearchResponse>("/search", {
        method: "POST",
        body: JSON.stringify(request),
      });
    } catch (err) {
      if (shouldRethrow(err)) throw err;
      return showcaseSearch(request);
    }
  },

  // Tokenization Inspector
  async inspectTokenization(
    request: TokenizationInspectRequest
  ): Promise<TokenizationInspectResponse> {
    try {
      return await fetchJson<TokenizationInspectResponse>("/tokenization/inspect", {
        method: "POST",
        body: JSON.stringify(request),
      });
    } catch (err) {
      if (shouldRethrow(err)) throw err;
      return showcaseTokenize(request);
    }
  },

  // Server-Sent Events stream for live experiment run tracking
  subscribeToExperimentStream(
    runId: string,
    onEvent: (event: StreamEvent) => void,
    onComplete?: () => void,
    onError?: (err: unknown) => void
  ): () => void {
    const url = `${API_BASE_URL}/experiments/${runId}/stream`;
    try {
      const eventSource = new EventSource(url);

      eventSource.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data) as StreamEvent;
          onEvent(parsed);
          if (parsed.stage === "COMPLETED" || parsed.stage === "FAILED") {
            eventSource.close();
            if (onComplete) onComplete();
          }
        } catch (e) {
          console.error("Failed to parse SSE event data", e);
        }
      };

      eventSource.onerror = (err) => {
        if (onError) onError(err);
        eventSource.close();
      };

      return () => {
        eventSource.close();
      };
    } catch {
      // In showcase mode, simulate immediate completion
      if (onComplete) onComplete();
      return () => {};
    }
  },
};
