"use client";

import * as React from "react";
import Link from "next/link";
import {
  FlaskConical,
  Play,
  ArrowRight,
  Clock,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Database,
} from "lucide-react";
import { api } from "@/lib/api";
import { Experiment } from "@/types";
import { formatDate } from "@/lib/utils";

export default function ExperimentsPage() {
  const [loading, setLoading] = React.useState(true);
  const [experiments, setExperiments] = React.useState<Experiment[]>([]);
  const [total, setTotal] = React.useState(0);
  const [limit, setLimit] = React.useState(10);
  const [offset, setOffset] = React.useState(0);
  const [error, setError] = React.useState<string | null>(null);

  const loadExperiments = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getExperiments(limit, offset);
      setExperiments(res.items);
      setTotal(res.total);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load experiments from gateway"
      );
    } finally {
      setLoading(false);
    }
  }, [limit, offset]);

  React.useEffect(() => {
    loadExperiments();
  }, [loadExperiments]);

  const totalPages = Math.max(1, Math.ceil(total / limit));
  const currentPage = Math.floor(offset / limit) + 1;

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      {/* Page Header */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-foreground">
          Experiment Matrix
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Multilingual Information Retrieval evaluation studies. Compare
          retrieval effectiveness across languages and models with paired
          statistical significance testing.
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-400 text-xs">
          {error}
        </div>
      )}

      {/* Experiments List Container */}
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <div className="p-5 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FlaskConical className="h-4 w-4 text-violet-500" />
            <span className="text-sm font-semibold text-foreground">
              Evaluated Studies
            </span>
            <span className="text-xs text-muted-foreground">({total} total)</span>
          </div>
        </div>

        {loading ? (
          <div className="p-12 text-center text-sm text-muted-foreground">
            Loading experiments from backend...
          </div>
        ) : experiments.length > 0 ? (
          <div className="divide-y divide-border">
            {experiments.map((exp) => {
              const latestRun =
                exp.runs && exp.runs.length > 0 ? exp.runs[0] : null;
              const status = latestRun ? latestRun.status : "CONFIGURED";

              return (
                <div
                  key={exp.id}
                  className="p-6 flex flex-col md:flex-row md:items-center justify-between gap-4 hover:bg-secondary/30 transition-colors"
                >
                  <div className="space-y-1.5 max-w-2xl">
                    <div className="flex items-center gap-2">
                      <Link
                        href={`/experiments/${exp.id}`}
                        className="text-base font-semibold text-foreground hover:text-primary transition"
                      >
                        {exp.name}
                      </Link>
                      <span
                        className={`px-2 py-0.5 text-[10px] font-semibold rounded uppercase tracking-wider ${
                          status === "COMPLETED"
                            ? "bg-emerald-500/10 text-emerald-500 border border-emerald-500/20"
                            : status === "FAILED"
                            ? "bg-rose-500/10 text-rose-500 border border-rose-500/20"
                            : "bg-amber-500/10 text-amber-500 border border-amber-500/20"
                        }`}
                      >
                        {status}
                      </span>
                    </div>

                    {exp.description && (
                      <p className="text-xs text-muted-foreground line-clamp-2">
                        {exp.description}
                      </p>
                    )}

                    <div className="flex flex-wrap items-center gap-3 pt-2 text-xs text-muted-foreground">
                      <span>Baseline: <strong>{exp.baseline_language_code}</strong></span>
                      <span>·</span>
                      <span>
                        Cutoffs: {exp.cutoff_k_values.map((k) => `@${k}`).join(", ")}
                      </span>
                      <span>·</span>
                      <span>Metric: {exp.similarity_metric}</span>
                      <span>·</span>
                      <span>Created: {formatDate(exp.created_at)}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    <Link
                      href={`/experiments/${exp.id}`}
                      className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg border border-border bg-secondary/60 text-xs font-medium text-foreground hover:bg-secondary hover:border-primary/40 transition"
                    >
                      View Report <ArrowRight className="h-3.5 w-3.5" />
                    </Link>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="p-12 text-center space-y-3">
            <FlaskConical className="h-10 w-10 text-muted-foreground mx-auto stroke-1" />
            <h4 className="text-sm font-semibold text-foreground">
              No Experiments Created Yet
            </h4>
            <p className="text-xs text-muted-foreground max-w-md mx-auto">
              Evaluation studies compute Precision@K, Recall@K, MRR, nDCG@K,
              and cross-lingual parity gaps. Check the backend tests or create
              an experiment using the API.
            </p>
          </div>
        )}

        {/* Pagination */}
        {total > limit && (
          <div className="p-4 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
            <div>
              Showing {offset + 1} to {Math.min(offset + limit, total)} of{" "}
              {total} experiments
            </div>
            <div className="flex items-center gap-2">
              <button
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - limit))}
                className="p-1.5 rounded border border-border bg-secondary/40 hover:bg-secondary disabled:opacity-40 transition"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <span className="px-2 font-mono">
                Page {currentPage} of {totalPages}
              </span>
              <button
                disabled={offset + limit >= total}
                onClick={() => setOffset(offset + limit)}
                className="p-1.5 rounded border border-border bg-secondary/40 hover:bg-secondary disabled:opacity-40 transition"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
