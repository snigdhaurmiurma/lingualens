"use client";

import * as React from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  Database,
  ArrowLeft,
  FileText,
  Hash,
  ShieldCheck,
  Search,
  Lock,
} from "lucide-react";
import { api } from "@/lib/api";
import { Dataset } from "@/types";
import { formatDate } from "@/lib/utils";

export default function DatasetDetailPage() {
  const params = useParams();
  const datasetId = params?.id as string;

  const [dataset, setDataset] = React.useState<Dataset | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    async function load() {
      if (!datasetId) return;
      setLoading(true);
      try {
        const data = await api.getDataset(datasetId);
        setDataset(data);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to load dataset details"
        );
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [datasetId]);

  return (
    <div className="space-y-6 max-w-5xl mx-auto pb-16">
      <Link
        href="/datasets"
        className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition"
      >
        <ArrowLeft className="h-3.5 w-3.5" /> Back to Datasets
      </Link>

      {loading ? (
        <div className="p-12 text-center text-sm text-muted-foreground">
          Loading dataset specifications...
        </div>
      ) : error ? (
        <div className="p-6 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-400 text-sm">
          {error}
        </div>
      ) : dataset ? (
        <div className="space-y-6">
          {/* Header Card */}
          <div className="rounded-xl border border-border bg-card p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-xl font-bold text-foreground">
                    {dataset.name}
                  </h2>
                  <span className="px-2 py-0.5 text-xs font-mono rounded bg-primary/10 text-primary border border-primary/20">
                    v{dataset.version}
                  </span>
                  {dataset.is_locked && (
                    <span className="px-2 py-0.5 text-[11px] rounded bg-secondary text-muted-foreground flex items-center gap-1">
                      <Lock className="h-3 w-3" /> Locked
                    </span>
                  )}
                </div>
                <p className="text-xs text-muted-foreground font-mono mt-1">
                  slug: {dataset.slug} · ID: {dataset.id}
                </p>
              </div>

              <Link
                href={`/query-lab?dataset=${dataset.id}`}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 transition"
              >
                <Search className="h-3.5 w-3.5" /> Query In Lab
              </Link>
            </div>

            {dataset.description && (
              <p className="text-sm text-muted-foreground leading-relaxed">
                {dataset.description}
              </p>
            )}

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-border">
              <div className="p-3 rounded-lg bg-secondary/30 border border-border">
                <div className="text-[11px] text-muted-foreground">
                  Documents
                </div>
                <div className="text-lg font-bold font-mono text-foreground">
                  {dataset.document_count}
                </div>
              </div>
              <div className="p-3 rounded-lg bg-secondary/30 border border-border">
                <div className="text-[11px] text-muted-foreground">
                  Queries
                </div>
                <div className="text-lg font-bold font-mono text-foreground">
                  {dataset.query_count}
                </div>
              </div>
              <div className="p-3 rounded-lg bg-secondary/30 border border-border">
                <div className="text-[11px] text-muted-foreground">
                  Query Groups
                </div>
                <div className="text-lg font-bold font-mono text-foreground">
                  {dataset.query_group_count}
                </div>
              </div>
              <div className="p-3 rounded-lg bg-secondary/30 border border-border">
                <div className="text-[11px] text-muted-foreground">
                  License
                </div>
                <div className="text-sm font-semibold text-foreground truncate mt-1">
                  {dataset.license}
                </div>
              </div>
            </div>

            {/* Integrity details */}
            <div className="pt-3 border-t border-border text-xs text-muted-foreground space-y-1 font-mono">
              <div className="flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-emerald-500" />
                <span>SHA256 Content Hash:</span>
                <span className="text-foreground truncate max-w-md">
                  {dataset.content_hash_sha256}
                </span>
              </div>
              <div>Created: {formatDate(dataset.created_at)}</div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
