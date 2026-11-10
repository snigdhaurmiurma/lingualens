"use client";

import * as React from "react";
import {
  Settings,
  Cpu,
  Database,
  Activity,
  HardDrive,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Terminal,
} from "lucide-react";
import { api } from "@/lib/api";
import { SystemInfoResponse, HealthResponse } from "@/types";

export default function SettingsPage() {
  const [systemInfo, setSystemInfo] = React.useState<SystemInfoResponse | null>(
    null
  );
  const [health, setHealth] = React.useState<HealthResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  const loadData = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [sys, h] = await Promise.all([
        api.getSystemInfo(),
        api.getHealth(),
      ]);
      setSystemInfo(sys);
      setHealth(h);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load system runtime metadata"
      );
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    loadData();
  }, [loadData]);

  const apiBaseUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";

  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-16">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-foreground">
            System &amp; Runtime Environment
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            Host machine specifications, PyTorch execution parameters, and
            vector storage engine configuration.
          </p>
        </div>
        <button
          onClick={loadData}
          disabled={loading}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-secondary/50 text-xs font-medium hover:bg-secondary transition"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh Runtime Info
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-400 text-xs">
          {error}
        </div>
      )}

      {loading ? (
        <div className="p-12 text-center text-sm text-muted-foreground">
          Querying host runtime parameters...
        </div>
      ) : (
        <div className="space-y-6">
          {/* Gateway & Network Configuration */}
          <div className="rounded-xl border border-border bg-card p-6 space-y-4">
            <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <Activity className="h-4 w-4 text-primary" />
              API Gateway Connectivity
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
              <div className="p-3 rounded-lg bg-secondary/30 border border-border">
                <span className="text-muted-foreground block mb-1">
                  Gateway URL (NEXT_PUBLIC_API_BASE_URL)
                </span>
                <span className="font-mono text-foreground font-semibold">
                  {apiBaseUrl}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-secondary/30 border border-border">
                <span className="text-muted-foreground block mb-1">
                  Health Check Status
                </span>
                <span className="font-mono text-emerald-500 font-semibold flex items-center gap-1.5">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  {health?.status || "Unknown"} (v{health?.version || "0.1.0"})
                </span>
              </div>
            </div>
          </div>

          {/* Inference & Hardware Configuration */}
          <div className="rounded-xl border border-border bg-card p-6 space-y-4">
            <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <Cpu className="h-4 w-4 text-emerald-500" />
              Host Compute &amp; PyTorch Engine
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-3 rounded-lg bg-secondary/30 border border-border">
                <span className="text-muted-foreground block text-[11px]">
                  Python Runtime
                </span>
                <span className="font-mono text-foreground font-semibold mt-1 block">
                  {systemInfo?.python_version?.split(" ")[0] || "3.11"}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-secondary/30 border border-border">
                <span className="text-muted-foreground block text-[11px]">
                  PyTorch Version
                </span>
                <span className="font-mono text-foreground font-semibold mt-1 block">
                  {systemInfo?.torch_version || "2.x"}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-secondary/30 border border-border">
                <span className="text-muted-foreground block text-[11px]">
                  Host CPU Cores
                </span>
                <span className="font-mono text-foreground font-semibold mt-1 block">
                  {systemInfo?.cpu_count || "—"} Cores
                </span>
              </div>
              <div className="p-3 rounded-lg bg-secondary/30 border border-border">
                <span className="text-muted-foreground block text-[11px]">
                  CPU Threads Allocated
                </span>
                <span className="font-mono text-primary font-semibold mt-1 block">
                  {systemInfo?.torch_cpu_threads || "—"} Threads
                </span>
              </div>
            </div>
          </div>

          {/* Vector Storage & Database Engine */}
          <div className="rounded-xl border border-border bg-card p-6 space-y-4">
            <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <Database className="h-4 w-4 text-sky-500" />
              Persistence &amp; Vector Index Layer
            </h3>
            <div className="p-4 rounded-lg bg-secondary/30 border border-border space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Storage Backend:</span>
                <span className="font-mono font-semibold text-foreground">
                  {health?.database === "connected"
                    ? "SQLite + NumPy Brute-Force (Development Persistence)"
                    : "Disconnected"}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed">
                Dual-mode vector retrieval supports exact NumPy brute-force
                cosine similarity and native PostgreSQL with the pgvector
                extension (`&lt;=&gt;` operator). The active mode can be toggled
                per search or experiment run.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
