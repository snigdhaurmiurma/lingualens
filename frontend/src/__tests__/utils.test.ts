import { describe, it, expect } from "vitest";
import {
  formatLatency,
  formatPercent,
  formatNumber,
  isRtlLanguage,
  getLanguageMeta,
  getLanguageFontClass,
  cn,
} from "../lib/utils";

describe("Frontend Utility Functions", () => {
  describe("formatLatency", () => {
    it("formats sub-millisecond latencies as < 1 ms", () => {
      expect(formatLatency(0.4)).toBe("< 1 ms");
      expect(formatLatency(0.0)).toBe("< 1 ms");
    });

    it("formats standard millisecond latencies", () => {
      expect(formatLatency(15.2)).toBe("15 ms");
      expect(formatLatency(450.8)).toBe("451 ms");
    });

    it("formats latencies >= 1000ms as seconds", () => {
      expect(formatLatency(1250)).toBe("1.25 s");
      expect(formatLatency(3000)).toBe("3.00 s");
    });

    it("handles null or undefined safely", () => {
      // @ts-expect-error testing null
      expect(formatLatency(null)).toBe("—");
      expect(formatLatency(NaN)).toBe("—");
    });
  });

  describe("formatPercent", () => {
    it("formats decimal ratio as percentage string", () => {
      expect(formatPercent(0.8523)).toBe("85.2%");
      expect(formatPercent(1.0)).toBe("100.0%");
      expect(formatPercent(0.0)).toBe("0.0%");
    });
  });

  describe("formatNumber", () => {
    it("formats numeric values with custom decimal precision", () => {
      expect(formatNumber(1234.567, 2)).toBe("1,234.57");
      expect(formatNumber(42, 0)).toBe("42");
    });
  });

  describe("Language Metadata & Script Helpers", () => {
    it("correctly identifies RTL for Arabic", () => {
      expect(isRtlLanguage("ar")).toBe(true);
      expect(isRtlLanguage("ar-SA")).toBe(true);
      expect(isRtlLanguage("en")).toBe(false);
      expect(isRtlLanguage("bn")).toBe(false);
      expect(isRtlLanguage("hi")).toBe(false);
      expect(isRtlLanguage("bn-Latn")).toBe(false);
    });

    it("returns proper metadata for registered languages", () => {
      const bn = getLanguageMeta("bn");
      expect(bn.label).toBe("Bengali");
      expect(bn.nativeName).toBe("বাংলা");
      expect(bn.script).toBe("Beng");

      const ar = getLanguageMeta("ar");
      expect(ar.label).toBe("Arabic");
      expect(ar.nativeName).toBe("العربية");
      expect(ar.isRtl).toBe(true);

      const banglish = getLanguageMeta("bn-Latn");
      expect(banglish.label).toBe("Banglish");
      expect(banglish.script).toBe("Latn");
    });

    it("assigns appropriate font classes for complex scripts", () => {
      expect(getLanguageFontClass("bn")).toBe("font-bengali");
      expect(getLanguageFontClass("hi")).toBe("font-devanagari");
      expect(getLanguageFontClass("ar")).toBe("font-arabic");
      expect(getLanguageFontClass("en")).toBe("font-sans");
      expect(getLanguageFontClass("bn-Latn")).toBe("font-sans");
    });
  });

  describe("ClassNames cn helper", () => {
    it("merges tailwind class names properly", () => {
      expect(cn("px-2", "py-1")).toBe("px-2 py-1");
      expect(cn("px-2", { "text-red-500": true, "bg-blue-500": false })).toBe(
        "px-2 text-red-500"
      );
    });
  });
});
