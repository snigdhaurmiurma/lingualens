import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { api, ApiError } from "../lib/api";

describe("LinguaLens API Client", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("fetches health status successfully", async () => {
    const mockHealth = {
      status: "healthy",
      version: "0.1.0",
      timestamp: "2026-11-10T00:00:00Z",
      database: "connected",
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockHealth,
    } as unknown as Response);

    const result = await api.getHealth();
    expect(result.status).toBe("healthy");
    expect(result.database).toBe("connected");
    expect(global.fetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/health",
      expect.objectContaining({
        headers: expect.objectContaining({
          "Content-Type": "application/json",
        }),
      })
    );
  });

  it("fetches registered embedding models", async () => {
    const mockModels = [
      {
        model_id: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        display_name: "MiniLM-L12",
        provider: "SentenceTransformers",
        parameter_count: 118000000,
        embedding_dimension: 384,
        max_sequence_length: 128,
        license: "Apache-2.0",
        query_prefix: "",
        document_prefix: "",
        supported_languages: ["en", "bn", "hi", "ar"],
        requires_gpu: false,
        is_loaded: true,
      },
    ];

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockModels,
    } as unknown as Response);

    const result = await api.getModels();
    expect(result).toHaveLength(1);
    expect(result[0].embedding_dimension).toBe(384);
  });

  it("executes semantic search with correct payload and method", async () => {
    const mockSearchResponse = {
      query_text: "জলবায়ু পরিবর্তন",
      model_id: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      retrieval_mode: "EXACT",
      top_k: 5,
      count: 1,
      latency_ms: 12.5,
      results: [
        {
          document_id: "11111111-1111-1111-1111-111111111111",
          doc_external_id: "doc-bn-01",
          score: 0.892,
          rank: 1,
          language_code: "bn",
          script_code: "Beng",
          content: "জলবায়ু পরিবর্তন একটি বৈশ্বিক সংকট।",
        },
      ],
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockSearchResponse,
    } as unknown as Response);

    const result = await api.search({
      query_text: "জলবায়ু পরিবর্তন",
      model_id: "minilm",
      top_k: 5,
    });

    expect(result.count).toBe(1);
    expect(result.results[0].score).toBe(0.892);
    expect(global.fetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/search",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          query_text: "জলবায়ু পরিবর্তন",
          model_id: "minilm",
          top_k: 5,
        }),
      })
    );
  });

  it("throws ApiError when backend returns 404 or 500 error", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({ detail: "Dataset not found" }),
    } as unknown as Response);

    await expect(api.getDataset("invalid-id")).rejects.toThrow(ApiError);
  });
});
