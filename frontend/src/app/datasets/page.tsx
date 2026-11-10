"use client";

import * as React from "react";
import Link from "next/link";
import {
  Database,
  Search,
  Plus,
  ArrowRight,
  FileText,
  HelpCircle,
  Hash,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
} from "lucide-react";
import { api } from "@/lib/api";
import { Dataset, PaginatedResponse } from "@/types";
import { formatDate } from "@/lib/utils";

export default function DatasetsPage() {
  const [loading, setLoading] = React.useState(true);
  const [datasets, setDatasets] = React.useState<Dataset[]>([]);
  const [total, setTotal] = React.useState(0);
  const [limit, setLimit] = React.useState(10);
  const [offset, setOffset] = React.useState(0);
  const [error, setError] = React.useState<string | null>(null);

  const loadDatasets = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getDatasets(limit, offset);
      setDatasets(res.items);
      setTotal(res.total);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load datasets from gateway"
      );
    } finally {
      setLoading(false);
    }
  }, [limit, offset]);

  React.useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  const totalPages = Math.max(1, Math.ceil(total / limit));
  const currentPage = Math.floor(offset / limit) + 1;

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-16">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-foreground">
            Dataset Studio
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            Multilingual benchmark corpora with parallel test queries, document
            chunks, and relevance judgments (qrels).
          </p>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-4 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-400 text-xs">
          {error}
        </div>
      )}

      {/* Dataset Table / Card List */}
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <div className="p-5 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold text-foreground">
              Corpus Collection
            </span>
            <span className="text-xs text-muted-foreground">({total} total)</span>
          </div>
        </div>

        {loading ? (
          <div className="p-12 text-center text-sm text-muted-foreground">
            Loading datasets from backend...
          </div>
        ) : datasets.length > 0 ? (
          <div className="divide-y divide-border">
            {datasets.map((ds) => (
              <div
                key={ds.id}
                className="p-6 flex flex-col md:flex-row md:items-center justify-between gap-4 hover:bg-secondary/30 transition-colors"
              >
                <div className="space-y-1.5 max-w-2xl">
                  <div className="flex items-center gap-2">
                    <Link
                      href={`/datasets/${ds.id}`}
                      className="text-base font-semibold text-foreground hover:text-primary transition"
                    >
                      {ds.name}
                    </Link>
                    <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-primary/10 text-primary border border-primary/20">
                      v{ds.version}
                    </span>
                    <span className="px-2 py-0.5 text-[10px] uppercase font-semibold rounded bg-secondary text-muted-foreground">
                      {ds.dataset_type}
                    </span>
                  </div>

                  {ds.description && (
                    <p className="text-xs text-muted-foreground line-clamp-2">
                      {ds.description}
                    </p>
                  )}

                  {/* Badges row */}
                  <div className="flex flex-wrap items-center gap-3 pt-2 text-xs text-muted-foreground">
                    <span className="font-mono">{ds.document_count} docs</span>
                    <span>·</span>
                    <span className="font-mono">{ds.query_count} queries</span>
                    <span>·</span>
                    <span>License: {ds.license}</span>
                    <span>·</span>
                    <span>Created: {formatDate(ds.created_at)}</span>
                  </div>

                  {/* Supported languages */}
                  <div className="flex items-center gap-1.5 pt-1">
                    <span className="text-[11px] text-muted-foreground">
                      Languages:
                    </span>
                    {ds.supported_languages.map((lang) => (
                      <span
                        key={lang}
                        className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-secondary/80 text-foreground border border-border"
                      >
                        {lang}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="flex items-center gap-3 shrink-0">
                  <Link
                    href={`/datasets/${ds.id}`}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-secondary/60 text-xs font-medium text-foreground hover:bg-secondary hover:border-primary/40 transition"
                  >
                    Inspect Details <ArrowRight className="h-3.5 w-3.5" />
                  </Link>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-12 text-center space-y-3">
            <Database className="h-10 w-10 text-muted-foreground mx-auto stroke-1" />
            <h4 className="text-sm font-semibold text-foreground">
              No Benchmark Datasets Found
            </h4>
            <p className="text-xs text-muted-foreground max-w-md mx-auto">
              The database currently contains no evaluation datasets. You can
              populate initial benchmark collections via the backend management
              scripts or through the API.
            </p>
          </div>
        )}

        {/* Pagination controls */}
        {total > limit && (
          <div className="p-4 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
            <div>
              Showing {offset + 1} to {Math.min(offset + limit, total)} of{" "}
              {total} datasets
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
