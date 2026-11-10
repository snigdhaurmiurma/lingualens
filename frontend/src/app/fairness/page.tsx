"use client";

import * as React from "react";
import Link from "next/link";
import {
  Scale,
  Sparkles,
  ArrowRight,
  Calculator,
  Layers,
  FlaskConical,
  BookOpen,
} from "lucide-react";

export default function FairnessPage() {
  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-16">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-violet-500/10 text-violet-400 border border-violet-500/20 mb-2">
          <Scale className="h-3.5 w-3.5" />
          Milestone 6 Architecture Roadmap
        </div>
        <h2 className="text-2xl font-bold tracking-tight text-foreground">
          Language Fairness &amp; Disparity Analyzer
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          Rigorous statistical evaluation of cross-lingual performance gaps,
          bootstrap 95% confidence intervals, and transliteration degradation.
        </p>
      </div>

      {/* Methodology Explanatory Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Card 1: Disparity Ratio */}
        <div className="rounded-xl border border-border bg-card p-6 space-y-4">
          <div className="flex items-center gap-2 text-primary font-semibold text-sm">
            <Calculator className="h-4 w-4" />
            Relative Disparity Ratio (RDR)
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Quantifies the proportion of retrieval effectiveness retained by a
            target non-English language relative to the English baseline:
          </p>
          <div className="p-3 rounded-lg bg-secondary/40 border border-border font-mono text-xs text-center text-foreground">
            RDR(L) = Metric(L) / Metric(Baseline_EN)
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            A ratio of 1.0 indicates perfect parity. Values significantly below
            1.0 (with paired p &lt; 0.05) indicate systemic multilingual
            disparity.
          </p>
        </div>

        {/* Card 2: Transliteration Penalty */}
        <div className="rounded-xl border border-border bg-card p-6 space-y-4">
          <div className="flex items-center gap-2 text-violet-400 font-semibold text-sm">
            <Layers className="h-4 w-4" />
            Transliteration Script Penalty (TSP)
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Evaluates the degradation introduced when native scripts are written
            using Latin transliteration (e.g. Banglish vs native Bengali):
          </p>
          <div className="p-3 rounded-lg bg-secondary/40 border border-border font-mono text-xs text-center text-foreground">
            TSP = (Metric_native - Metric_translit) / Metric_native
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Directly highlights subword token fragmentation and vocabulary bias
            in multilingual subword tokenizers.
          </p>
        </div>
      </div>

      {/* Integration Notice */}
      <div className="rounded-xl border border-dashed border-border bg-card/40 p-8 text-center space-y-3">
        <FlaskConical className="h-8 w-8 text-muted-foreground mx-auto stroke-1" />
        <h4 className="text-sm font-semibold text-foreground">
          Fairness Analyzer Implementation Status
        </h4>
        <p className="text-xs text-muted-foreground max-w-lg mx-auto">
          The mathematical core (Milestone 1) is implemented and verified in the
          backend. Dedicated interactive disparity visualizations and bootstrap
          distribution charts are scheduled for Milestone 6 following frontend
          foundation review.
        </p>
        <div className="pt-2">
          <Link
            href="/experiments"
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-secondary text-secondary-foreground text-xs font-medium hover:bg-secondary/80 transition border border-border"
          >
            Review Experiment Matrix <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </div>
    </div>
  );
}
