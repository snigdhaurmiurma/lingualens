"use client";

import * as React from "react";
import Link from "next/link";
import { FileBarChart, Download, FileText, Code2, ArrowRight } from "lucide-react";

export default function ReportsPage() {
  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-16">
      <div>
        <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-primary/10 text-primary border border-primary/20 mb-2">
          <FileBarChart className="h-3.5 w-3.5" />
          Milestone 7 Architecture Roadmap
        </div>
        <h2 className="text-2xl font-bold tracking-tight text-foreground">
          Research &amp; Evaluation Reports
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Automated publication-ready research reports, LaTeX camera-ready
          tables, and machine-readable JSON/CSV export packages.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="rounded-xl border border-border bg-card p-6 space-y-3">
          <FileText className="h-5 w-5 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">
            LaTeX Table Generator
          </h3>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Exports formatted tabular environments with confidence interval
            subscripts and statistical significance asterisks (* p &lt; 0.05).
          </p>
        </div>

        <div className="rounded-xl border border-border bg-card p-6 space-y-3">
          <Download className="h-5 w-5 text-sky-400" />
          <h3 className="text-sm font-semibold text-foreground">
            Reproducibility Bundles
          </h3>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Packages full run configurations, random seeds, hardware metadata,
            and raw query rankings into a standardized zip archive.
          </p>
        </div>

        <div className="rounded-xl border border-border bg-card p-6 space-y-3">
          <Code2 className="h-5 w-5 text-emerald-400" />
          <h3 className="text-sm font-semibold text-foreground">
            JSON / CSV Export
          </h3>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Structured schemas suitable for integration with downstream Jupyter
            notebooks, pandas pipelines, and external BI dashboards.
          </p>
        </div>
      </div>

      <div className="rounded-xl border border-dashed border-border bg-card/40 p-8 text-center space-y-3">
        <h4 className="text-sm font-semibold text-foreground">
          Reporting Engine Milestone Status
        </h4>
        <p className="text-xs text-muted-foreground max-w-lg mx-auto">
          Exporting and report generation are planned for Milestone 7. Current
          evaluation summaries can be inspected interactively in the Experiments
          Matrix.
        </p>
        <div className="pt-2">
          <Link
            href="/experiments"
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-secondary text-secondary-foreground text-xs font-medium hover:bg-secondary/80 transition border border-border"
          >
            Go to Experiments Matrix <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </div>
    </div>
  );
}
