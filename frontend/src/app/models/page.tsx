"use client";

import * as React from "react";
import Link from "next/link";
import { Boxes, Cpu, ArrowRight, ShieldCheck, Tag, Terminal } from "lucide-react";
import { api } from "@/lib/api";
import { ModelInfo } from "@/types";

export default function ModelsPage() {
  const [models, setModels] = React.useState<ModelInfo[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const data = await api.getModels();
        setModels(data);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to load models from gateway"
        );
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-foreground">
          Model Adapter Registry
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Registered multilingual dense embedding models with standardized
          query prefixes, subword tokenizers, and CPU vector inference.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-400 text-xs">
          {error}
        </div>
      )}

      {loading ? (
        <div className="p-12 text-center text-sm text-muted-foreground">
          Querying model adapters from gateway...
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {models.map((model) => (
            <div
              key={model.model_id}
              className="rounded-xl border border-border bg-card p-6 space-y-5 hover:border-primary/40 transition-colors"
            >
              <div className="space-y-1">
                <div className="flex items-start justify-between gap-2">
                  <h3 className="text-base font-semibold text-foreground">
                    {model.display_name}
                  </h3>
                  <span className="px-2 py-0.5 text-[10px] uppercase font-bold rounded bg-primary/10 text-primary border border-primary/20">
                    {model.license}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground font-mono">
                  {model.model_id}
                </p>
              </div>

              {/* Specs Grid */}
              <div className="grid grid-cols-3 gap-3 p-3 rounded-lg bg-secondary/30 border border-border text-center">
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    Embedding Dim
                  </div>
                  <div className="text-base font-bold font-mono text-foreground mt-0.5">
                    {model.embedding_dimension}d
                  </div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    Parameters
                  </div>
                  <div className="text-base font-bold font-mono text-foreground mt-0.5">
                    {(model.parameter_count / 1_000_000).toFixed(0)}M
                  </div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    Max Context
                  </div>
                  <div className="text-base font-bold font-mono text-foreground mt-0.5">
                    {model.max_sequence_length} tokens
                  </div>
                </div>
              </div>

              {/* Prefix details */}
              <div className="space-y-2 text-xs">
                <div className="flex items-center justify-between p-2 rounded bg-secondary/40 border border-border">
                  <span className="text-muted-foreground">Query Prefix:</span>
                  <span className="font-mono text-foreground font-semibold">
                    {model.query_prefix ? `"${model.query_prefix}"` : "(none)"}
                  </span>
                </div>
                <div className="flex items-center justify-between p-2 rounded bg-secondary/40 border border-border">
                  <span className="text-muted-foreground">Document Prefix:</span>
                  <span className="font-mono text-foreground font-semibold">
                    {model.document_prefix
                      ? `"${model.document_prefix}"`
                      : "(none)"}
                  </span>
                </div>
              </div>

              {/* Supported Languages */}
              <div className="space-y-1.5">
                <span className="text-[11px] text-muted-foreground font-medium">
                  Reported Language Coverage:
                </span>
                <div className="flex flex-wrap gap-1">
                  {model.supported_languages.map((l) => (
                    <span
                      key={l}
                      className="px-2 py-0.5 text-[10px] font-mono rounded bg-secondary text-foreground border border-border"
                    >
                      {l}
                    </span>
                  ))}
                </div>
              </div>

              <div className="pt-2 border-t border-border flex items-center justify-between">
                <span className="text-xs text-muted-foreground flex items-center gap-1.5">
                  <Cpu className="h-3.5 w-3.5 text-emerald-500" />
                  CPU-Optimized Inference
                </span>
                <Link
                  href={`/query-lab?model=${encodeURIComponent(model.model_id)}`}
                  className="inline-flex items-center gap-1 text-xs font-semibold text-primary hover:underline"
                >
                  Test in Query Lab <ArrowRight className="h-3 w-3" />
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
