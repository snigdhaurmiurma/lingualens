"use client";

import * as React from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  FlaskConical,
  Play,
  ArrowLeft,
  CheckCircle2,
  AlertCircle,
  Clock,
  Sparkles,
  Scale,
  Activity,
  Layers,
  BarChart3,
} from "lucide-react";
import { api } from "@/lib/api";
import {
  Experiment,
  ExperimentRun,
  ExperimentEvaluationSummary,
  StreamEvent,
  RetrievalMode,
} from "@/types";
import { formatDate, formatPercent, getLanguageMeta } from "@/lib/utils";

export default function ExperimentDetailPage() {
  const params = useParams();
  const expId = params?.id as string;

  const [experiment, setExperiment] = React.useState<Experiment | null>(null);
  const [summary, setSummary] =
    React.useState<ExperimentEvaluationSummary | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  // Run execution controls
  const [runModel, setRunModel] = React.useState(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  );
  const [runMode, setRunMode] = React.useState<RetrievalMode>("EXACT");
  const [isRunning, setIsRunning] = React.useState(false);
  const [activeRunId, setActiveRunId] = React.useState<string | null>(null);
  const [streamProgress, setStreamProgress] = React.useState<StreamEvent | null>(
    null
  );

  const loadData = React.useCallback(async () => {
    if (!expId) return;
    setLoading(true);
    setError(null);
    try {
      const expData = await api.getExperiment(expId);
      setExperiment(expData);

      // Check if there are completed runs, load summary
      try {
        const sumData = await api.getExperimentSummary(expId);
        setSummary(sumData);
      } catch {
        // Summary might not exist yet if no completed runs
        setSummary(null);
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load experiment"
      );
    } finally {
      setLoading(false);
    }
  }, [expId]);

  React.useEffect(() => {
    loadData();
  }, [loadData]);

  // Handle run trigger
  async function handleTriggerRun() {
    if (!expId) return;
    setIsRunning(true);
    setStreamProgress(null);

    try {
      const run = await api.runExperiment(expId, {
        model_id: runModel,
        retrieval_mode: runMode,
      });

      setActiveRunId(run.id);

      // Subscribe to SSE stream for live updates
      const unsubscribe = api.subscribeToExperimentStream(
        run.id,
        (ev) => {
          setStreamProgress(ev);
        },
        () => {
          // Complete
          setIsRunning(false);
          loadData();
        },
        (err) => {
          console.error("SSE error", err);
          setIsRunning(false);
          loadData();
        }
      );
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to trigger experiment run"
      );
      setIsRunning(false);
    }
  }

  return (
    <div className="space-y-8 max-w-6xl mx-auto pb-16">
      <Link
        href="/experiments"
        className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition"
      >
        <ArrowLeft className="h-3.5 w-3.5" /> Back to Experiments
      </Link>

      {error && (
        <div className="p-4 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-400 text-xs">
          {error}
        </div>
      )}

      {loading ? (
        <div className="p-12 text-center text-sm text-muted-foreground">
          Loading experiment specification and runs...
        </div>
      ) : experiment ? (
        <div className="space-y-8">
          {/* Header Card */}
          <div className="rounded-xl border border-border bg-card p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h2 className="text-xl font-bold text-foreground">
                  {experiment.name}
                </h2>
                <p className="text-xs text-muted-foreground font-mono mt-0.5">
                  ID: {experiment.id} · Dataset: {experiment.dataset_id}
                </p>
              </div>

              <div className="flex items-center gap-2">
                <span className="px-2.5 py-1 text-xs font-semibold rounded bg-secondary text-foreground">
                  Baseline: <strong>{experiment.baseline_language_code}</strong>
                </span>
                <span className="px-2.5 py-1 text-xs font-mono rounded bg-primary/10 text-primary border border-primary/20">
                  {experiment.similarity_metric}
                </span>
              </div>
            </div>

            {experiment.description && (
              <p className="text-sm text-muted-foreground leading-relaxed">
                {experiment.description}
              </p>
            )}

            {/* Evaluation cutoffs */}
            <div className="pt-2 flex items-center gap-2 text-xs text-muted-foreground">
              <span>Evaluation Depths:</span>
              {experiment.cutoff_k_values.map((k) => (
                <span
                  key={k}
                  className="px-2 py-0.5 rounded bg-secondary text-foreground font-mono text-xs"
                >
                  @{k}
                </span>
              ))}
            </div>
          </div>

          {/* Run Trigger & Real-time SSE Progress Container */}
          <div className="rounded-xl border border-border bg-card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-foreground">
                  Trigger Evaluation Run
                </h3>
                <p className="text-xs text-muted-foreground">
                  Execute full corpus encoding, dual-mode retrieval, and paired
                  significance testing
                </p>
              </div>
            </div>

            {/* Run Controls */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label className="text-[11px] font-medium text-muted-foreground mb-1 block">
                  Embedding Model
                </label>
                <select
                  disabled={isRunning}
                  value={runModel}
                  onChange={(e) => setRunModel(e.target.value)}
                  className="w-full px-2.5 py-1.5 text-xs rounded-lg border border-border bg-secondary/50 text-foreground"
                >
                  <option value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2">
                    MiniLM-L12-v2 (384d)
                  </option>
                  <option value="intfloat/multilingual-e5-small">
                    Multilingual-E5-Small (384d)
                  </option>
                </select>
              </div>

              <div>
                <label className="text-[11px] font-medium text-muted-foreground mb-1 block">
                  Retrieval Mode
                </label>
                <select
                  disabled={isRunning}
                  value={runMode}
                  onChange={(e) => setRunMode(e.target.value as RetrievalMode)}
                  className="w-full px-2.5 py-1.5 text-xs rounded-lg border border-border bg-secondary/50 text-foreground"
                >
                  <option value="EXACT">EXACT (Cosine similarity)</option>
                  <option value="VECTOR_INDEX">VECTOR_INDEX (pgvector)</option>
                </select>
              </div>

              <div className="flex items-end">
                <button
                  type="button"
                  disabled={isRunning}
                  onClick={handleTriggerRun}
                  className="w-full py-2 px-4 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:bg-primary/90 disabled:opacity-50 transition flex items-center justify-center gap-1.5 shadow-sm shadow-primary/20"
                >
                  <Play className={`h-3.5 w-3.5 ${isRunning ? "animate-spin" : ""}`} />
                  {isRunning ? "Running Study..." : "Execute Run"}
                </button>
              </div>
            </div>

            {/* Live SSE Stream Progress Indicator */}
            {streamProgress && (
              <div className="p-4 rounded-lg border border-primary/30 bg-primary/5 space-y-2 mt-4">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-primary flex items-center gap-2">
                    <Activity className="h-3.5 w-3.5 animate-pulse" />
                    Stage: {streamProgress.stage}
                  </span>
                  <span className="font-mono text-foreground font-semibold">
                    {streamProgress.percent}%
                  </span>
                </div>

                {/* Progress Bar */}
                <div className="w-full h-2 rounded-full bg-secondary overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${Math.min(100, streamProgress.percent)}%` }}
                  />
                </div>

                <p className="text-[11px] text-muted-foreground font-mono truncate">
                  {streamProgress.detail}
                </p>
              </div>
            )}
          </div>

          {/* Evaluation Results Section */}
          {summary && summary.metrics && summary.metrics.length > 0 ? (
            <div className="space-y-6">
              {/* IR Metrics Table */}
              <div className="rounded-xl border border-border bg-card overflow-hidden">
                <div className="p-5 border-b border-border flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-semibold text-foreground">
                      Language IR Performance Matrix
                    </h3>
                    <p className="text-xs text-muted-foreground">
                      Mean performance with 95% bootstrap confidence intervals
                      and paired significance vs {summary.baseline_language}
                    </p>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-secondary/40 text-muted-foreground uppercase text-[10px] font-semibold tracking-wider border-b border-border">
                      <tr>
                        <th className="px-4 py-3">Language</th>
                        <th className="px-4 py-3">Metric</th>
                        <th className="px-4 py-3 text-right">Value</th>
                        <th className="px-4 py-3 text-center">95% CI</th>
                        <th className="px-4 py-3 text-right">p-value</th>
                        <th className="px-4 py-3 text-center">Significance</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {summary.metrics.map((m, idx) => {
                        const langMeta = getLanguageMeta(m.language_code);
                        return (
                          <tr
                            key={idx}
                            className="hover:bg-secondary/20 transition-colors"
                          >
                            <td className="px-4 py-3 font-semibold text-foreground">
                              {m.language_code} ({langMeta.label})
                            </td>
                            <td className="px-4 py-3 font-mono text-muted-foreground">
                              {m.metric_name}
                            </td>
                            <td className="px-4 py-3 text-right font-mono font-bold text-foreground">
                              {m.metric_value.toFixed(4)}
                            </td>
                            <td className="px-4 py-3 text-center font-mono text-muted-foreground">
                              {typeof m.ci_lower_95 === "number" && typeof m.ci_upper_95 === "number"
                                ? `[${m.ci_lower_95.toFixed(3)}, ${m.ci_upper_95.toFixed(3)}]`
                                : "—"}
                            </td>
                            <td className="px-4 py-3 text-right font-mono text-muted-foreground">
                              {typeof m.p_value_vs_baseline === "number"
                                ? m.p_value_vs_baseline.toFixed(4)
                                : "—"}
                            </td>
                            <td className="px-4 py-3 text-center">
                              {m.is_statistically_significant ? (
                                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/10 text-amber-500 border border-amber-500/20">
                                  p &lt; 0.05 *
                                </span>
                              ) : (
                                <span className="text-[10px] text-muted-foreground">
                                  ns
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Language Fairness Disparity Table */}
              {summary.fairness_disparities &&
                summary.fairness_disparities.length > 0 && (
                  <div className="rounded-xl border border-border bg-card overflow-hidden">
                    <div className="p-5 border-b border-border">
                      <h3 className="text-sm font-semibold text-foreground">
                        Cross-Lingual Fairness Disparities
                      </h3>
                      <p className="text-xs text-muted-foreground">
                        Absolute performance gap and relative degradation
                        compared to baseline ({summary.baseline_language})
                      </p>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-xs text-left">
                        <thead className="bg-secondary/40 text-muted-foreground uppercase text-[10px] font-semibold tracking-wider border-b border-border">
                          <tr>
                            <th className="px-4 py-3">Evaluated Language</th>
                            <th className="px-4 py-3">Reference Baseline</th>
                            <th className="px-4 py-3 text-right">
                              Performance Gap
                            </th>
                            <th className="px-4 py-3 text-right">
                              Disparity Ratio
                            </th>
                            <th className="px-4 py-3 text-center">
                              Significant Disparity
                            </th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                          {summary.fairness_disparities.map((fd, idx) => (
                            <tr
                              key={idx}
                              className="hover:bg-secondary/20 transition-colors"
                            >
                              <td className="px-4 py-3 font-semibold text-foreground">
                                {fd.language_code}
                              </td>
                              <td className="px-4 py-3 text-muted-foreground font-mono">
                                {fd.baseline_language}
                              </td>
                              <td className="px-4 py-3 text-right font-mono font-bold text-foreground">
                                {fd.absolute_performance_gap.toFixed(4)}
                              </td>
                              <td className="px-4 py-3 text-right font-mono text-muted-foreground">
                                {formatPercent(fd.relative_disparity_ratio)}
                              </td>
                              <td className="px-4 py-3 text-center">
                                {fd.is_statistically_significant ? (
                                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/10 text-rose-500 border border-rose-500/20">
                                    Significant Gap
                                  </span>
                                ) : (
                                  <span className="text-[10px] text-muted-foreground">
                                    Parity Retained
                                  </span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-border bg-card/40 p-12 text-center space-y-3">
              <FlaskConical className="h-8 w-8 text-muted-foreground mx-auto stroke-1" />
              <p className="text-sm font-semibold text-foreground">
                No Evaluation Results Available Yet
              </p>
              <p className="text-xs text-muted-foreground max-w-sm mx-auto">
                Click &ldquo;Execute Run&rdquo; above to start the semantic
                encoding and retrieval pipeline. Once complete, evaluation
                metrics and bootstrap confidence intervals will display here.
              </p>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
