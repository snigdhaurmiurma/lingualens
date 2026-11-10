"use client";

import * as React from "react";
import { usePathname } from "next/navigation";
import {
  Sun,
  Moon,
  Activity,
  CheckCircle2,
  XCircle,
  RefreshCw,
  ExternalLink,
} from "lucide-react";
import { useTheme } from "@/components/theme-provider";
import { api } from "@/lib/api";
import { HealthResponse } from "@/types";

const ROUTE_NAMES: Record<string, string> = {
  "/": "Overview Dashboard",
  "/query-lab": "Multilingual Query Lab",
  "/datasets": "Dataset Studio",
  "/experiments": "Experiment Matrix",
  "/models": "Model Adapter Registry",
  "/fairness": "Language Fairness Analyzer",
  "/reports": "Research & Evaluation Reports",
  "/settings": "System & Runtime Environment",
};

export function Header() {
  const pathname = usePathname();
  const { theme, toggleTheme } = useTheme();
  const [health, setHealth] = React.useState<HealthResponse | null>(null);
  const [isChecking, setIsChecking] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  const checkHealth = React.useCallback(async () => {
    setIsChecking(true);
    try {
      const data = await api.getHealth();
      setHealth(data);
      setError(null);
    } catch (err) {
      setHealth(null);
      setError("Gateway unreachable");
    } finally {
      setIsChecking(false);
    }
  }, []);

  React.useEffect(() => {
    checkHealth();
    // Poll every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  const currentTitle =
    ROUTE_NAMES[pathname] ||
    (pathname.startsWith("/datasets/")
      ? "Dataset Inspection"
      : pathname.startsWith("/experiments/")
      ? "Experiment Inspection"
      : "LinguaLens");

  return (
    <header className="h-16 border-b border-border bg-card/40 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-30">
      {/* Route Title */}
      <div className="flex items-center gap-3">
        <h1 className="text-base font-semibold text-foreground tracking-tight">
          {currentTitle}
        </h1>
      </div>

      {/* Action / Status Pill */}
      <div className="flex items-center gap-3">
        {/* Gateway Health Indicator */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-border bg-secondary/50 text-xs font-medium">
          {isChecking && !health ? (
            <Activity className="h-3.5 w-3.5 text-muted-foreground animate-pulse" />
          ) : health && health.status === "healthy" ? (
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
          ) : (
            <XCircle className="h-3.5 w-3.5 text-rose-500" />
          )}

          <span className="hidden sm:inline">
            {isChecking && !health
              ? "Checking Gateway..."
              : health && health.status === "healthy"
              ? "Gateway Connected"
              : "Gateway Offline"}
          </span>

          {health && (
            <span className="text-[10px] font-mono text-muted-foreground border-l border-border pl-1.5 ml-0.5">
              DB: {health.database}
            </span>
          )}

          <button
            onClick={() => checkHealth()}
            title="Refresh Gateway Connection"
            className="text-muted-foreground hover:text-foreground transition-colors ml-1"
          >
            <RefreshCw
              className={`h-3 w-3 ${isChecking ? "animate-spin" : ""}`}
            />
          </button>
        </div>

        {/* Theme Toggle Button */}
        <button
          onClick={toggleTheme}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          className="p-2 rounded-md border border-border bg-secondary/40 text-muted-foreground hover:text-foreground hover:bg-secondary/80 transition-colors"
        >
          {theme === "dark" ? (
            <Sun className="h-4 w-4 text-amber-400" />
          ) : (
            <Moon className="h-4 w-4 text-slate-700" />
          )}
        </button>
      </div>
    </header>
  );
}
