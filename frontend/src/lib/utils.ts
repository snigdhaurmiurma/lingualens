import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { SUPPORTED_LANGUAGES, LanguageCode } from "@/types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(num: number, decimals: number = 2): string {
  if (num === null || num === undefined || isNaN(num)) return "—";
  return num.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function formatPercent(num: number, decimals: number = 1): string {
  if (num === null || num === undefined || isNaN(num)) return "—";
  return `${(num * 100).toFixed(decimals)}%`;
}

export function formatLatency(ms: number): string {
  if (ms === null || ms === undefined || isNaN(ms)) return "—";
  if (ms < 1) return "< 1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatDate(dateStr: string): string {
  if (!dateStr) return "—";
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return dateStr;
  }
}

export function getLanguageMeta(code: LanguageCode) {
  const match = SUPPORTED_LANGUAGES.find((lang) => lang.code === code);
  if (match) return match;
  return {
    code,
    label: code.toUpperCase(),
    nativeName: code,
    script: "Latn",
    isRtl: false,
  };
}

export function isRtlLanguage(code: LanguageCode): boolean {
  return code === "ar" || code.startsWith("ar-");
}

export function getLanguageFontClass(code: LanguageCode): string {
  if (code === "bn") return "font-bengali";
  if (code === "hi") return "font-devanagari";
  if (code === "ar") return "font-arabic";
  return "font-sans";
}
