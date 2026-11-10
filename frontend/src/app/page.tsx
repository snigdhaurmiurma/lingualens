"use client";

import * as React from "react";
import Link from "next/link";
import {
  Boxes,
  Database,
  FlaskConical,
  Cpu,
  Search,
  ArrowRight,
  CheckCircle2,
  Clock,
  AlertTriangle,
  Layers,
  Sparkles,
  ExternalLink,
  ShieldCheck,
} from "lucide-react";
import { api } from "@/lib/api";
import {
  ModelInfo,
  Dataset,
  Experiment,
  SystemInfoResponse,
  HealthResponse,
} from "@/types";
import { formatDate } from "@/lib/utils";

export default function OverviewPage() {
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  const [models, setModels] = React.useState<ModelInfo[]>([]);
  const [datasetsTotal, setDatasetsTotal] = React.useState<number>(0);
  const [datasets, setDatasets] = React.useState<Dataset[]>([]);
  const [experimentsTotal, setExperimentsTotal] = React.useState<number>(0);
  const [experiments, setExperiments] = React.useState<Experiment[]>([]);
  const [systemInfo, setSystemInfo] = React.useState<SystemInfoResponse | null>(
    null
  );
  const [health, setHealth] = React.useState<HealthResponse | null>(null);

  React.useEffect(() => {
    async function loadDashboardData() {
      setLoading(true);
      setError(null);

      try {
        const [modelsRes, datasetsRes, experimentsRes, sysRes, healthRes] =
          await Promise.allSettled([
            api.getModels(),
            api.getDatasets(5, 0),
            api.getExperiments(5, 0),
            api.getSystemInfo(),
            api.getHealth(),
          ]);

        if (modelsRes.status === "fulfilled") setModels(modelsRes.value);
        if (datasetsRes.status === "fulfilled") {
          setDatasetsTotal(datasetsRes.value.total);
          setDatasets(datasetsRes.value.items);
        }
        if (experimentsRes.status === "fulfilled") {
          setExperimentsTotal(experimentsRes.value.total);
          setExperiments(experimentsRes.value.items);
        }
        if (sysRes.status === "fulfilled") setSystemInfo(sysRes.value);
        if (healthRes.status === "fulfilled") setHealth(healthRes.value);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to connect to LinguaLens gateway"
        );
      } finally {
        setLoading(false);
      }
    }

    loadDashboardData();
  }, []);

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Hero / Platform Overview */}
      <div className="relative overflow-hidden rounded-xl border border-border bg-gradient-to-br from-card via-card to-primary/5 p-8 shadow-sm">
        <div className="max-w-3xl space-y-3">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-semibold bg-primary/10 text-primary border border-primary/20">
            <Sparkles className="h-3.5 w-3.5" />
            Empirical Evaluation Framework
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
            Multilingual Search Fairness & Retrieval Disparity
          </h2>
          <p className="text-sm sm:text-base text-muted-foreground leading-relaxed">
            Quantify semantic retrieval performance, cross-lingual degradation,
            and transliteration penalties across English, Bengali, Banglish,
            Hindi, and Arabic with rigorous statistical significance.
          </p>
          <div className="pt-2 flex flex-wrap gap-3">
            <Link
              href="/query-lab"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition shadow-sm shadow-primary/25"
            >
              <Search className="h-4 w-4" />
              Open Multilingual Query Lab
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/datasets"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-secondary text-secondary-foreground text-sm font-medium hover:bg-secondary/80 transition border border-border"
            >
              <Database className="h-4 w-4" />
              Explore Benchmark Datasets
            </Link>
          </div>
        </div>
      </div>

      {/* KPI Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric 1: Models */}
        <div className="rounded-lg border border-border bg-card p-5 space-y-2">
          <div className="flex items-center justify-between text-muted-foreground">
            <span className="text-xs font-medium uppercase tracking-wider">
              Registered Models
            </span>
            <Boxes className="h-4 w-4 text-primary" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold tracking-tight text-foreground">
              {loading ? "…" : models.length}
            </span>
            <span className="text-xs text-muted-foreground">in registry</span>
          </div>
          <p className="text-xs text-muted-foreground">
            MiniLM-L12 &amp; Multilingual-E5
          </p>
        </div>

        {/* Metric 2: Datasets */}
        <div className="rounded-lg border border-border bg-card p-5 space-y-2">
          <div className="flex items-center justify-between text-muted-foreground">
            <span className="text-xs font-medium uppercase tracking-wider">
              Corpus Datasets
            </span>
            <Database className="h-4 w-4 text-sky-500" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold tracking-tight text-foreground">
              {loading ? "…" : datasetsTotal}
            </span>
            <span className="text-xs text-muted-foreground">benchmarks</span>
          </div>
          <p className="text-xs text-muted-foreground">
            Parallel multilingual queries &amp; qrels
          </p>
        </div>

        {/* Metric 3: Experiments */}
        <div className="rounded-lg border border-border bg-card p-5 space-y-2">
          <div className="flex items-center justify-between text-muted-foreground">
            <span className="text-xs font-medium uppercase tracking-wider">
              Experiments
            </span>
            <FlaskConical className="h-4 w-4 text-violet-500" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold tracking-tight text-foreground">
              {loading ? "…" : experimentsTotal}
            </span>
            <span className="text-xs text-muted-foreground">studies</span>
          </div>
          <p className="text-xs text-muted-foreground">
            With bootstrap 95% CIs &amp; p-values
          </p>
        </div>

        {/* Metric 4: Hardware / Inference Engine */}
        <div className="rounded-lg border border-border bg-card p-5 space-y-2">
          <div className="flex items-center justify-between text-muted-foreground">
            <span className="text-xs font-medium uppercase tracking-wider">
              Execution Engine
            </span>
            <Cpu className="h-4 w-4 text-emerald-500" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-xl font-bold tracking-tight text-foreground uppercase">
              {systemInfo?.device_default || "CPU-OPT"}
            </span>
            <span className="text-xs text-emerald-500 font-medium">
              {systemInfo ? `${systemInfo.torch_cpu_threads} Threads` : "Ready"}
            </span>
          </div>
          <p className="text-xs text-muted-foreground truncate">
            PyTorch {systemInfo?.torch_version || "2.x"} · NumPy Exact
          </p>
        </div>
      </div>

      {/* Model Registry Cards */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-foreground tracking-tight">
              Model Adapter Registry
            </h3>
            <p className="text-xs text-muted-foreground">
              Loaded embedding adapters with prompt prefixing and subword
              tokenizers
            </p>
          </div>
          <Link
            href="/models"
            className="text-xs font-medium text-primary hover:underline flex items-center gap-1"
          >
            View Specs <ArrowRight className="h-3 w-3" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {models.length > 0 ? (
            models.map((model) => (
              <div
                key={model.model_id}
                className="rounded-lg border border-border bg-card p-6 flex flex-col justify-between space-y-4 hover:border-primary/40 transition-colors"
              >
                <div className="space-y-2">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <h4 className="font-semibold text-foreground text-sm">
                        {model.display_name}
                      </h4>
                      <p className="text-xs text-muted-foreground font-mono mt-0.5">
                        {model.model_id}
                      </p>
                    </div>
                    <span className="px-2 py-0.5 text-[10px] uppercase font-bold rounded bg-primary/10 text-primary border border-primary/20">
                      {model.license}
                    </span>
                  </div>

                  <p className="text-xs text-muted-foreground">
                    Provider: <span className="text-foreground">{model.provider}</span>
                  </p>
                </div>

                {/* Specs Badges */}
                <div className="grid grid-cols-3 gap-2 py-3 border-y border-border/80 text-center">
                  <div>
                    <div className="text-[11px] text-muted-foreground">Dim</div>
                    <div className="text-sm font-semibold font-mono text-foreground">
                      {model.embedding_dimension}
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] text-muted-foreground">
                      Params
                    </div>
                    <div className="text-sm font-semibold font-mono text-foreground">
                      {(model.parameter_count / 1_000_000).toFixed(0)}M
                    </div>
                  </div>
                  <div>
                    <div className="text-[11px] text-muted-foreground">
                      Context
                    </div>
                    <div className="text-sm font-semibold font-mono text-foreground">
                      {model.max_sequence_length} toks
                    </div>
                  </div>
                </div>

                {/* Query Prefix Indicator */}
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">
                    Query Prefix:{" "}
                    <code className="text-primary font-mono font-semibold">
                      {model.query_prefix ? `"${model.query_prefix}"` : "None"}
                    </code>
                  </span>
                  <Link
                    href={`/query-lab?model=${encodeURIComponent(model.model_id)}`}
                    className="text-xs font-medium text-foreground hover:text-primary flex items-center gap-1"
                  >
                    Test Query <ArrowRight className="h-3 w-3" />
                  </Link>
                </div>
              </div>
            ))
          ) : (
            <div className="col-span-2 rounded-lg border border-dashed border-border p-8 text-center text-muted-foreground">
              {loading
                ? "Connecting to model registry..."
                : "No models reported by gateway. Ensure backend is running."}
            </div>
          )}
        </div>
      </div>

      {/* Two Column Layout: Recent Experiments & Benchmark Datasets */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Recent Experiments Card */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-semibold text-foreground tracking-tight">
                Recent Experiments
              </h3>
              <p className="text-xs text-muted-foreground">
                Statistical evaluation studies and runs
              </p>
            </div>
            <Link
              href="/experiments"
              className="text-xs font-medium text-primary hover:underline flex items-center gap-1"
            >
              All Experiments <ArrowRight className="h-3 w-3" />
            </Link>
          </div>

          <div className="rounded-lg border border-border bg-card divide-y divide-border">
            {experiments.length > 0 ? (
              experiments.map((exp) => {
                const latestRun = exp.runs && exp.runs.length > 0 ? exp.runs[0] : null;
                const status = latestRun ? latestRun.status : "CONFIGURED";

                return (
                  <div
                    key={exp.id}
                    className="p-4 flex items-center justify-between gap-4 hover:bg-secondary/30 transition-colors"
                  >
                    <div className="min-w-0 space-y-1">
                      <Link
                        href={`/experiments/${exp.id}`}
                        className="font-medium text-sm text-foreground hover:text-primary truncate block"
                      >
                        {exp.name}
                      </Link>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <span>Baseline: {exp.baseline_language_code}</span>
                        <span>·</span>
                        <span>{formatDate(exp.created_at)}</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
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
                      <Link
                        href={`/experiments/${exp.id}`}
                        className="p-1 rounded text-muted-foreground hover:text-foreground"
                      >
                        <ArrowRight className="h-4 w-4" />
                      </Link>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="p-8 text-center space-y-2">
                <FlaskConical className="h-8 w-8 text-muted-foreground mx-auto stroke-1" />
                <p className="text-sm font-medium text-foreground">
                  No evaluation experiments recorded yet
                </p>
                <p className="text-xs text-muted-foreground max-w-sm mx-auto">
                  Execute your first cross-lingual IR study to compute
                  Precision@K, Recall@K, MRR, nDCG@K, and paired p-values.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Benchmark Datasets Card */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-semibold text-foreground tracking-tight">
                Benchmark Datasets
              </h3>
              <p className="text-xs text-muted-foreground">
                Indexed multilingual evaluation corpora
              </p>
            </div>
            <Link
              href="/datasets"
              className="text-xs font-medium text-primary hover:underline flex items-center gap-1"
            >
              Browse Corpora <ArrowRight className="h-3 w-3" />
            </Link>
          </div>

          <div className="rounded-lg border border-border bg-card divide-y divide-border">
            {datasets.length > 0 ? (
              datasets.map((ds) => (
                <div
                  key={ds.id}
                  className="p-4 flex items-center justify-between gap-4 hover:bg-secondary/30 transition-colors"
                >
                  <div className="min-w-0 space-y-1">
                    <Link
                      href={`/datasets/${ds.id}`}
                      className="font-medium text-sm text-foreground hover:text-primary truncate block"
                    >
                      {ds.name}
                    </Link>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <span>{ds.document_count} documents</span>
                      <span>·</span>
                      <span>{ds.query_count} queries</span>
                      <span>·</span>
                      <span className="uppercase text-[10px] font-mono px-1 rounded bg-secondary">
                        {ds.dataset_type}
                      </span>
                    </div>
                  </div>

                  <Link
                    href={`/datasets/${ds.id}`}
                    className="p-1 rounded text-muted-foreground hover:text-foreground shrink-0"
                  >
                    <ArrowRight className="h-4 w-4" />
                  </Link>
                </div>
              ))
            ) : (
              <div className="p-8 text-center space-y-2">
                <Database className="h-8 w-8 text-muted-foreground mx-auto stroke-1" />
                <p className="text-sm font-medium text-foreground">
                  No benchmark datasets found
                </p>
                <p className="text-xs text-muted-foreground max-w-sm mx-auto">
                  Create or ingest evaluation corpora with parallel multilingual
                  queries to begin semantic indexing.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
