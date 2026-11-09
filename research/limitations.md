# LinguaLens: Methodological Limitations & Research Boundaries

**Document Version:** 1.0.0  
**Status:** Research Transparency Disclosure  
**Date:** November 2026  

---

## 1. Construct Validity & Evaluation Scope

### 1.1 Intent Equivalence Assumption
LinguaLens evaluates cross-lingual disparity under the construct of **Paired Semantic Query Groups** ($Q_G$). This methodology assumes that an information need can be translated across languages while preserving an identical set of relevant corpus documents. 
- *Limitation:* In culturally specific or jurisdictionally bounded domains (such as local legal statutes, vernacular idioms, or regional healthcare practices), identical informational needs may not exist or may map to divergent real-world expectations.

### 1.2 Binary Relevance vs. Multi-Faceted Satisfaction
While LinguaLens implements graded relevance ($r \in \{0, 1, 2, 3\}$), nDCG metrics evaluate top-K ordering based on static relevance labels. Real-world search satisfaction encompasses document freshness, readability, linguistic fluency, and presentation layout, which are outside the scope of vector similarity rankings.

---

## 2. Orthographic & Script Instability

### 2.1 Banglish Orthographic Permutations
Informal Latin-script Bengali ("Banglish") lacks an official or standardized orthography. A single native Bengali phrase such as:
> **"আপনি কেমন আছেন?"** (How are you?)

can be rendered in Latin script in multiple phonetic permutations:
1. `apni kemon achen?`
2. `apni kmn achen?`
3. `apni kemn asen?`
4. `apni kemon asen?`

- *Limitation:* If an evaluation dataset tests only one canonical transliteration, it may either underestimate or overestimate model robustness. LinguaLens addresses this by incorporating orthographic perturbation suites into future benchmark revisions.

---

## 3. Pretraining Contamination & Dataset Leakage

Many prominent multilingual benchmark datasets (e.g., subsets derived from Wikipedia or Common Crawl) may have been included in the self-supervised pretraining corpora (mC4, OSCAR, RedPajama) of modern language models.
- *Risk:* A model may demonstrate high retrieval performance on Bengali Wikipedia articles not because of superior cross-lingual semantic alignment, but due to verbatim memorization during pretraining.
- *Mitigation in LinguaLens:* LinguaLens mandates date-stamped benchmark provenance and explicitly labels contemporary benchmark sets sourced post-cutoff from regional news and community QA archives.

---

## 4. Disparity vs. Quality Decoupling

A critical tenet of fairness research:
$$\text{Low Disparity} \not\equiv \text{High Quality}$$

If a model scores $nDCG@10 = 0.12$ on English and $nDCG@10 = 0.11$ on Bengali, its Absolute Disparity is remarkably low ($\Delta_{abs} = 0.01$). However, the retrieval system is uniformly dysfunctional. LinguaLens reports both absolute quality ($nDCG@K$) and relative disparity ($D_{rel}$) side-by-side to prevent misleading conclusions.

---

## 5. Latency Measurement Variability

Latency benchmarks executed on shared or non-isolated CPU hardware (e.g., cloud VMs with burstable vCPUs or background OS tasks) are prone to jitter.
- *Guideline:* For publication-grade latency benchmarking, users must execute runs on pinned dedicated CPU cores or dedicated GPU instances with CPU frequency governors set to maximum performance.
