/**
 * Typed API Client for LinguaLens Backend Gateway
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

export const api = {
  // System & Health
  async getHealth(): Promise<HealthResponse> {
    return fetchJson<HealthResponse>("/health");
  },

  async getSystemInfo(): Promise<SystemInfoResponse> {
    return fetchJson<SystemInfoResponse>("/system/info");
  },

  // Embedding Models
  async getModels(): Promise<ModelInfo[]> {
    return fetchJson<ModelInfo[]>("/models");
  },

  // Datasets
  async getDatasets(
    limit = 50,
    offset = 0
  ): Promise<PaginatedResponse<Dataset>> {
    return fetchJson<PaginatedResponse<Dataset>>(
      `/datasets?limit=${limit}&offset=${offset}`
    );
  },

  async getDataset(id: string): Promise<Dataset> {
    return fetchJson<Dataset>(`/datasets/${id}`);
  },

  // Experiments
  async getExperiments(
    limit = 50,
    offset = 0
  ): Promise<PaginatedResponse<Experiment>> {
    return fetchJson<PaginatedResponse<Experiment>>(
      `/experiments?limit=${limit}&offset=${offset}`
    );
  },

  async getExperiment(id: string): Promise<Experiment> {
    return fetchJson<Experiment>(`/experiments/${id}`);
  },

  async getExperimentSummary(
    id: string
  ): Promise<ExperimentEvaluationSummary> {
    return fetchJson<ExperimentEvaluationSummary>(
      `/experiments/${id}/summary`
    );
  },

  async runExperiment(
    experimentId: string,
    params: {
      model_id?: string;
      retrieval_mode?: string;
      random_seed?: number;
    } = {}
  ): Promise<ExperimentRun> {
    return fetchJson<ExperimentRun>(`/experiments/${experimentId}/run`, {
      method: "POST",
      body: JSON.stringify(params),
    });
  },

  // Semantic Search
  async search(request: SearchRequest): Promise<SearchResponse> {
    return fetchJson<SearchResponse>("/search", {
      method: "POST",
      body: JSON.stringify(request),
    });
  },

  // Tokenization Inspector
  async inspectTokenization(
    request: TokenizationInspectRequest
  ): Promise<TokenizationInspectResponse> {
    return fetchJson<TokenizationInspectResponse>("/tokenization/inspect", {
      method: "POST",
      body: JSON.stringify(request),
    });
  },

  // Server-Sent Events stream for live experiment run tracking
  subscribeToExperimentStream(
    runId: string,
    onEvent: (event: StreamEvent) => void,
    onComplete?: () => void,
    onError?: (err: unknown) => void
  ): () => void {
    const url = `${API_BASE_URL}/experiments/${runId}/stream`;
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
  },
};
