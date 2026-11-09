# LinguaLens: Research & Evaluation Methodology

**Document Version:** 1.0.0  
**Scientific Review:** Completed  
**Domain:** Cross-Lingual Information Retrieval & Algorithmic Fairness  
**Date:** November 2026  

---

## 1. Theoretical Framework

Modern dense retrieval models use dual-encoders (bi-encoders) to map queries $q$ and documents $d$ into a shared dense representation space $\mathbb{R}^d$:

$$s(q, d) = \cos(\mathbf{e}_q, \mathbf{e}_d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\| \|\mathbf{e}_d\|}$$

While cross-lingual encoders are trained on parallel or translated corpora (such as OPUS or Wikipedia bitexts), representation spaces often suffer from **representation anisotropy** and **script-bound topological distortion**. In particular, queries expressing identical semantic intent $\mathcal{I}$ across different languages $\mathcal{L} = \{en, bn, bn\text{-}Latn, hi, ar\}$ do not map to equidistant locations relative to relevant corpus documents.

LinguaLens isolates and measures this disparity without conflating it with the overall retrieval capacity of the model.

---

## 2. Information Retrieval Metrics

Let $\mathcal{Q}$ be the set of queries. For a given query $q \in \mathcal{Q}$, let $R_q(k)$ denote the document retrieved at rank $k \in [1, K]$, and let $r(q, d) \in \{0, 1, 2, 3\}$ be the graded relevance judgment:
- **0:** Completely Irrelevant
- **1:** Marginally Relevant / Topical overlap without answering query
- **2:** Relevant / Satisfies substantial portion of intent
- **3:** Highly Relevant / Definitively answers query

### 2.1 Precision@K and Recall@K
Under binary binarization with threshold $\tau = 2$ (documents with $r(q, d) \ge 2$ considered relevant):

$$\text{Precision@K}(q) = \frac{1}{K} \sum_{k=1}^K \mathbb{I}[r(q, R_q(k)) \ge \tau]$$

$$\text{Recall@K}(q) = \frac{\sum_{k=1}^K \mathbb{I}[r(q, R_q(k)) \ge \tau]}{|\{d \in \mathcal{D} : r(q, d) \ge \tau\}|}$$

### 2.2 Mean Reciprocal Rank (MRR)
MRR rewards systems that position the first relevant document at the top of the SERP:

$$\text{MRR}(q) = \frac{1}{\min \{k : r(q, R_q(k)) \ge \tau\} \cup \{\infty\}}$$

$$\text{MRR}(\mathcal{Q}) = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \text{MRR}(q)$$

### 2.3 Normalized Discounted Cumulative Gain (nDCG@K)
nDCG@K fully exploits graded relevance judgments by exponentially rewarding high relevance and logarithmically discounting lower ranks:

$$\text{DCG@K}(q) = \sum_{k=1}^K \frac{2^{r(q, R_q(k))} - 1}{\log_2(k + 1)}$$

Let $\pi^*$ be the ideal ranking permutation sorting all documents in corpus $\mathcal{D}$ in descending order of true relevance $r(q, d)$. The Ideal Discounted Cumulative Gain is:

$$\text{IDCG@K}(q) = \sum_{k=1}^{\min(K, |\mathcal{D}|)} \frac{2^{r(q, \pi^*(k))} - 1}{\log_2(k + 1)}$$

$$\text{nDCG@K}(q) = \begin{cases} \frac{\text{DCG@K}(q)}{\text{IDCG@K}(q)} & \text{if } \text{IDCG@K}(q) > 0 \\ 0 & \text{otherwise} \end{cases}$$

---

## 3. Cross-Lingual Fairness & Disparity Formulation

To measure disparity strictly attributable to language or script, LinguaLens structures evaluations around **Paired Query Groups** ($Q_G$).

### 3.1 Paired Semantic Query Groups
A Query Group $g \in \mathcal{G}$ represents an invariant semantic need:

$$Q_G(g) = \{q_{g, L} : L \in \mathcal{L}\}$$

where $q_{g, L}$ is the query translated and localized into language/script $L$. Crucially, the ground truth relevance vector $\{r(q_{g, L}, d)\}_{d \in \mathcal{D}}$ is held invariant across all language representations within the group:

$$\forall L_1, L_2 \in \mathcal{L}, \quad r(q_{g, L_1}, d) = r(q_{g, L_2}, d)$$

### 3.2 Disparity Metrics
Let $M(q)$ be an evaluation metric (e.g., $nDCG@10$) for query $q$. Let $L_{\text{base}}$ be the baseline language (typically English, $en$).

1. **Absolute Performance Gap ($\Delta_{abs}$):**
   $$\Delta_{abs}(L, L_{\text{base}}) = \frac{1}{|\mathcal{G}|} \sum_{g \in \mathcal{G}} \left[ M(q_{g, L_{\text{base}}}) - M(q_{g, L}) \right]$$
   - $\Delta_{abs} > 0$: Disadvantage against target language $L$.
   - $\Delta_{abs} \approx 0$: Parity.
   - $\Delta_{abs} < 0$: Superior performance on language $L$.

2. **Relative Disparity Ratio ($D_{rel}$):**
   $$D_{rel}(L, L_{\text{base}}) = \frac{\sum_{g \in \mathcal{G}} M(q_{g, L})}{\sum_{g \in \mathcal{G}} M(q_{g, L_{\text{base}}}) + \epsilon}$$

3. **Transliteration Penalty ($\Phi_{\text{trans}}$):**
   Measures the specific degradation incurred when users switch from native script to Latin phonetics (e.g., Bangla in Bengali script vs. Banglish in Latin script):
   $$\Phi_{\text{trans}}(bn) = \frac{1}{|\mathcal{G}|} \sum_{g \in \mathcal{G}} \left[ M(q_{g, bn\text{-Beng}}) - M(q_{g, bn\text{-Latn}}) \right]$$

---

## 4. Statistical Rigor & Hypothesis Testing

### 4.1 Bootstrap Confidence Intervals
Because retrieval metric distributions are bounded in $[0, 1]$ and often highly skewed (non-Gaussian), standard parametric errors underestimate uncertainty. LinguaLens implements the non-parametric percentile bootstrap:

1. Draw $B = 1000$ bootstrap resamples with replacement of query groups $\mathcal{G}^*_b \subseteq \mathcal{G}$.
2. Compute the metric $M_b$ for each sample.
3. Compute the empirical 95% confidence interval:
   $$CI_{0.95} = [M^*_{(\lfloor 0.025 \cdot B \rfloor)}, M^*_{(\lceil 0.975 \cdot B \rceil)}]$$

### 4.2 Non-Parametric Significance Testing (Paired Wilcoxon Signed-Rank Test)
To verify whether observed inter-language performance disparity is statistically genuine or merely an artifact of random query sampling variance, LinguaLens applies the **Paired Wilcoxon Signed-Rank Test** across paired metric differences:

$$D_g = M(q_{g, L_{\text{base}}}) - M(q_{g, L}), \quad \forall g \in \mathcal{G}$$

#### Hypotheses:
- **Null Hypothesis ($H_0$):** The distribution of paired query metric differences $D_g$ is symmetric about zero (i.e., median difference is zero; there is no language-level advantage).
- **Alternative Hypothesis ($H_1$):** The distribution of differences is shifted away from zero (two-sided test).

#### Core Methodological Assumptions:
1. **Genuinely Paired Observations:** The samples are strictly paired: each pair $(M(q_{g, L_{\text{base}}}), M(q_{g, L}))$ originates from the identical semantic query group $g$, holding the information need and corpus ground truth invariant.
2. **Independent Query Groups:** Each semantic query group $g_i$ is sampled independently from other query groups $g_j$ ($i \ne j$).
3. **Continuous/Ordinal Scale:** Differences $D_g \in [-1.0, 1.0]$ have an ordered numerical scale.
4. **Symmetry Under $H_0$:** The distribution of differences under the null hypothesis is assumed symmetric. **Crucially, normality is not assumed**, making this test robust for bounded, non-Gaussian IR metrics like $nDCG$ and $MRR$.

#### Scientific Interpretation of the $p$-Value:
The $p$-value represents the conditional probability:
$$p = P\left( |W| \ge |W_{\text{obs}}| \;\middle|\; H_0 \text{ is true} \right)$$
- If $p < \alpha$ (where $\alpha = 0.05$ is the pre-specified significance threshold), we **reject the null hypothesis $H_0$** at significance level $\alpha$. This indicates that under the assumption of equal language retrieval performance, an inter-language gap this large would occur by chance less than 5% of the time.
- **Critical Distinction:** A $p$-value is *not* the probability that the alternative hypothesis is true, nor does it measure the magnitude of the disparity (the magnitude is quantified by $\Delta_{abs}$ and its 95% bootstrap confidence interval). LinguaLens always presents the effect size ($\Delta_{abs}$), its empirical 95% CI, and the $p$-value jointly.

---

## 5. Benchmarking Dataset Schema

Every benchmark dataset must provide tabular/JSONL records conforming strictly to:

```json
{
  "query_id": "q-0104-bn",
  "query_group_id": "qg-0104",
  "query_text": "মৌসুমি সর্দি ও কাশির লক্ষণ",
  "language_code": "bn",
  "script_code": "Beng",
  "document_id": "doc-8911",
  "document_text": "মৌসুমি সর্দি সাধারণত রাইনোভাইরাস দ্বারা সৃষ্ট...",
  "relevance_judgment": 3,
  "dataset_version": "1.0.0",
  "provenance": "HUMAN_VERIFIED"
}
```

---

## 6. Scientific Integrity Directives

1. **Synthetic Data Quarantine:** Synthetic or machine-translated queries must never be presented as empirical ground truth. They must carry the label `SYNTHETIC_UNCORRECTED` until validated by native speakers.
2. **Determinism:** Seed initialization (`torch.manual_seed`, `np.random.seed`) must be logged for every run.
3. **Equal Metrics $\ne$ Fairness:** Equal nDCG scores across languages do not guarantee semantic fairness if error types vary (e.g., catastrophic misretrievals in health/legal domains). Qualitative error analyses are documented alongside numerical tables.
