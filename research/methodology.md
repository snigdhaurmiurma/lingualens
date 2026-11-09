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

---

## 7. Subword Fertility & Tokenizer Fragmentation

A tokenizer converts surface text strings into integer token sequences for neural model consumption. Multilingual models employ subword tokenizers (e.g., SentencePiece / Byte-Pair Encoding or Unigram) with fixed vocabulary sizes (e.g., 250,000 for XLM-RoBERTa). Because vocabulary allocation is heavily biased towards dominant pre-training corpora (English, Romance languages), underrepresented or morphologically rich languages suffer from severe subword fragmentation.

### 7.1 Corrected Subword Fertility Ratio Formula
Subword fertility measures how many subword units are required on average to represent a single natural word. **Crucially, special structural boundary tokens (such as sequence start `<s>` and sequence end `</s>`) are metadata artifacts of the transformer architecture, not linguistic subwords.** Including boundary tokens artificially inflates fertility, especially for short queries.

Therefore, LinguaLens defines Subword Fertility strictly over natural linguistic subword units:

$$\text{Fertility}(T) = \frac{|\mathcal{T}_{\text{linguistic}}(T)|}{|\mathcal{W}(T)|}$$

where:
- $\mathcal{W}(T)$ is the sequence of natural words obtained via whitespace/punctuation tokenization.
- $\mathcal{T}_{\text{linguistic}}(T)$ is the sequence of subword tokens produced by the model tokenizer, **excluding all special boundary tokens** ($\{ \texttt{<s>}, \texttt{</s>}, \texttt{[CLS]}, \texttt{[SEP]} \}$).

#### Validated Analytical Fertility Across Languages (MiniLM-L12-v2 / XLM-R Tokenizer):
- **English:** 5 linguistic tokens / 4 words = **1.25**
- **Bengali (Native Script):** 8 linguistic tokens / 3 words = **2.67**
- **Banglish (Latin Transliteration):** 6 linguistic tokens / 3 words = **2.00**
- **Hindi (Devanagari):** 6 linguistic tokens / 4 words = **1.50**
- **Arabic (Arabic Script):** 8 linguistic tokens / 3 words = **2.67**

### 7.2 Disentangling Tokenizer Fragmentation from Semantic Understanding
It is a common misconception in multilingual NLP to assume that high token fragmentation directly causes or equates to poor semantic retrieval:
1. **Fragmentation $\ne$ Incomprehension:** A high fertility ratio indicates that the model must compose meaning across more individual embedding vectors, which places higher demand on the transformer's self-attention mechanism and consumes more context window budget. However, if the transformer's self-attention layers have observed sufficient co-occurrence patterns during pre-training, it can still compose highly accurate semantic representations from fragmented subwords.
2. **Compact Tokens $\ne$ Semantic Alignment:** Conversely, a language or script with low subword fertility may still suffer from poor semantic alignment if the parallel training corpora did not adequately ground those concepts in cross-lingual space.
3. **Research Implication:** Tokenizer fragmentation is an *input efficiency and representation load metric*, whereas IR metrics ($nDCG$, Precision) and Cosine Similarity are *downstream semantic output metrics*. LinguaLens measures both to analyze whether and where fragmentation correlates with retrieval degradation, without confusing correlation with causation.

---

## 8. Cross-Lingual Semantic Space & Similarity Dynamics

### 8.1 Cosine Similarity vs. Downstream Retrieval Effectiveness
In a bi-encoder architecture, query and passage vectors are compared via cosine similarity. While cross-lingual cosine similarity between parallel sentence pairs indicates semantic vector alignment in the embedding space:
- High cosine similarity between parallel queries indicates geometric closeness, but **does not guarantee high retrieval rank** if corpus passages are pulled closer to conflicting distractor documents.
- Moderate cosine similarity between parallel queries across languages can still yield identical top-$K$ retrieval rankings if the target documents are similarly positioned relative to both language queries.

### 8.2 Granular Linguistic Sensitivity in Cross-Lingual Embeddings
Empirical evaluations demonstrate that cross-lingual cosine similarity is highly sensitive to query category, syntactic domain, and vocabulary borrowing:
- **General Knowledge / Common Concepts:** Parallel concepts with clear cross-lingual equivalents exhibit strong alignment in MiniLM-L12-v2:
  - *"what is the capital of France?"* $\leftrightarrow$ *"ফ্রান্সের রাজধানী কী?"*: $\cos = \mathbf{0.8355}$
  - *"climate change and global warming"* $\leftrightarrow$ *"জলবায়ু পরিবর্তন এবং বৈশ্বিক উষ্ণতা"*: $\cos = \mathbf{0.6760}$
  - *"water is essential for life"* $\leftrightarrow$ *"পানি জীবনের জন্য অপরিহার্য"*: $\cos = \mathbf{0.6104}$
  - *"the sun rises in the east"* $\leftrightarrow$ *"সূর্য পূর্ব দিকে ওঠে"*: $\cos = \mathbf{0.5794}$
- **Domain-Specific / Transliterated Loanwords:** When a query contains transliterated medical or technical terms (e.g., *"symptoms of seasonal flu"* vs. *"মৌসুমি ফ্লুর লক্ষণ"*, where *"ফ্লু"* is an English loanword phonetically transliterated into Bengali script), cross-lingual similarity can drop markedly (e.g., $0.1300$).
- **Methodological Standard:** LinguaLens enforces that low similarity on an isolated single query or query group must **never** be cited as evidence of overall language-family failure. Reliable conclusions require broad, multi-domain benchmark sets ($N \ge 100$ query groups) spanning natural queries, technical terminology, and colloquial expressions.

---

## 9. Asymmetric Retrieval & Model-Specific Instruction Prefixes

Dense retrieval models vary in their training objectives:
- **Symmetric Encoders (e.g., MiniLM-L12-v2):** Trained using sentence-pair classification and paraphrase scoring. Query and document texts are encoded symmetrically without prefixes.
- **Asymmetric Encoders (e.g., E5 - Weakly-Supervised Contrastive Pre-training):** Specifically trained to differentiate short search queries from full document passages.
  - Queries **must** be prepended with `"query: "` (e.g., `"query: symptoms of seasonal flu"`).
  - Documents / Passages **must** be prepended with `"passage: "` (e.g., `"passage: Influenza is a viral infection..."`).
  - Omitting these prefixes causes a distribution shift that severely harms cross-lingual retrieval accuracy and violates the model's design specifications. LinguaLens adapters automatically manage these prefixes transparently via `encode_queries()` and `encode_documents()`.

---

## 10. Empirical CPU Inference Profiling

To guarantee accessibility for developers and researchers without dedicated GPU clusters, LinguaLens profiles and optimizes all embedding adapters for multi-threaded CPU execution.

Empirical micro-benchmarking on an Intel Core i5/i7 equivalent laptop environment with 2 PyTorch CPU threads yielded the following baseline metrics for `paraphrase-multilingual-MiniLM-L12-v2` (117M parameters, 384 dimensions):

| Metric | Measurement | Description |
| :--- | :--- | :--- |
| **PyTorch CPU Threads** | `2` | Configured via `torch.set_num_threads()` |
| **Cold Model Load Time** | `~32.7 s` | Initial disk read, tensor allocation, and Hugging Face weight hydration |
| **Cold First Inference** | `266.37 ms` | Single query forward pass including first-time kernel dispatch |
| **Warm Query Latency (Mean)** | `103.44 ms` | Average inference time across 10 sequential warm queries |
| **Warm Query Latency (p50)** | `99.88 ms` | Median single-query response time |
| **Warm Query Latency (p95)** | `131.71 ms` | 95th percentile single-query response time |
| **Batch Size 1 Latency** | `106.58 ms` | `106.58 ms` / query |
| **Batch Size 5 Latency** | `200.85 ms` | `40.17 ms` / query (**2.65x throughput speedup**) |
| **Batch Size 10 Latency** | `267.98 ms` | `26.80 ms` / query (**3.98x throughput speedup**) |

*Key Insight:* Increasing the inference batch size from 1 to 10 delivers a ~4x throughput increase per query on CPU by exploiting vectorized matrix multiplication (BLAS) across transformer attention heads without increasing memory footprint significantly.

---

## 11. Vector Retrieval Taxonomy & Evaluation Standards

To maintain scientific precision, LinguaLens strictly demarcates vector search accuracy from information retrieval effectiveness:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Retrieval Evaluation Layers                     │
├────────────────────────────────────┬───────────────────────────────────┤
│    Vector Space Approximation      │     Information Retrieval Task    │
│            Fidelity                │            Effectiveness          │
├────────────────────────────────────┼───────────────────────────────────┤
│ Metric: IndexRecall@K              │ Metric: IR Recall@K, nDCG@K       │
│ Baseline: Exact Brute-Force Vector │ Baseline: Human Relevance Truth   │
│ Measures: Graph approximation loss │ Measures: Semantic understanding  │
└────────────────────────────────────┴───────────────────────────────────┘
```

### 11.1 Exact vs. Approximate Nearest-Neighbor Search
1. **Exact Nearest-Neighbor Search (Exhaustive Brute-Force):**
   - Computes pairwise cosine distance against every indexed document vector in the corpus ($\mathcal{O}(N)$ computational complexity).
   - Guarantees **$100\%$ Search Fidelity** ($IndexRecall@K = 1.0$), meaning it retrieves the true top-$K$ geometrically closest vectors in the dense embedding space.
   - **CRITICAL SCIENTIFIC DISTINCTION:** Exact nearest-neighbor retrieval does **NOT** guarantee $100\%$ Information Retrieval Recall ($\text{IR Recall@K}$). If an embedding model fails to place relevant documents in close proximity to the query vector, exact retrieval will faithfully retrieve irrelevant documents that happen to be close in vector space. Exact retrieval isolates the *embedding model's semantic quality* by eliminating indexing approximation error.
2. **Approximate Nearest-Neighbor (ANN) Search (e.g., HNSW, IVFFlat):**
   - Employs heuristic proximity graphs or Voronoi clustering to avoid full-corpus scans.
   - Accelerates retrieval from linear time to sub-linear empirical latency, but trades search fidelity: graph pruning may fail to visit the true geometric nearest neighbors.

### 11.2 Disambiguation of Evaluation Metrics
1. **Index Fidelity Metric ($\text{IndexRecall@K}$):**
   - Evaluates the vector index structure relative to the exact brute-force baseline:
     $$\text{IndexRecall@K} = \frac{|\mathcal{R}_{\text{approx@K}} \cap \mathcal{R}_{\text{exact@K}}|}{K}$$
   - Where $\mathcal{R}_{\text{exact@K}}$ is the set of top-$K$ documents returned by exhaustive brute-force search, and $\mathcal{R}_{\text{approx@K}}$ is the set returned by the approximate index.
2. **Information Retrieval Task Metric ($\text{IR Recall@K}$):**
   - Evaluates whether the retrieved documents satisfy the human information need:
     $$\text{IR Recall@K} = \frac{|\{d \in \mathcal{R}_{@K} : r(q, d) \ge \tau\}|}{|\{d \in \mathcal{D} : r(q, d) \ge \tau\}|}$$
   - Where $r(q, d) \in \{0, 1, 2, 3\}$ is the ground-truth relevance judgment annotated by human experts, and $\tau = 2$ is the relevance threshold.

---

## 12. HNSW Index Architecture & Configuration in PostgreSQL pgvector

Hierarchical Navigable Small World (HNSW) is a multi-layer proximity graph structure where upper layers contain long-range skip edges and lower layers contain densely connected local neighborhoods.

### 12.1 Index Creation & Operator Class
In PostgreSQL + `pgvector`, HNSW is constructed using the cosine distance operator `<=>`:

```sql
CREATE INDEX idx_document_embeddings_hnsw
ON document_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

- **Distance Operator:** `<=>` computes Cosine Distance:
  $$\text{dist}_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  $$\text{sim}_{\cos}(\mathbf{u}, \mathbf{v}) = 1.0 - \text{dist}_{\cos}(\mathbf{u}, \mathbf{v})$$
- **Operator Class:** `vector_cosine_ops` binds the HNSW graph traversal specifically to cosine distance.

### 12.2 Tuning Parameters
- **`m` (Default: 16):** Maximum number of bi-directional connections per graph node. Higher `m` improves index recall and graph connectivity on clustered multilingual datasets, at the expense of higher index build time and memory usage.
- **`ef_construction` (Default: 64):** Size of the dynamic candidate list evaluated during index insertion. A larger `ef_construction` produces a higher quality graph topology.
- **`ef_search` (Default: 40):** Size of the dynamic candidate list maintained during query execution:
  ```sql
  SET hnsw.ef_search = 40;
  ```
  Adjustable dynamically at query time: increasing `ef_search` increases `IndexRecall@K` towards $1.0$ at the cost of higher query latency.

### 12.3 Complexity Clarification
While theoretical small-world graph routing exhibits $\mathcal{O}(\log N)$ hops under uniform metric spaces, real-world high-dimensional vector spaces (such as $d = 384$ or $1024$) suffer from the **curse of dimensionality**, hubness, and representation anisotropy. HNSW must therefore be characterized as an **approximate graph-based retrieval algorithm with empirical sub-linear search characteristics**, rather than a mathematically guaranteed $\mathcal{O}(\log N)$ bound.

---

## 13. Embedding Vector Dimension Strategy

### 13.1 Current Implementation (Fixed 384d Baseline)
In Milestone 2 and 3, LinguaLens prioritizes lightweight CPU inference using two 384-dimensional models:
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` ($d = 384$)
- `intfloat/multilingual-e5-small` ($d = 384$)

In the current PostgreSQL schema:
- The column is defined via `VectorType(dim=384)`, which compiles to `vector(384)` in PostgreSQL.
- **Dimension Validation:**
  - Application level: `DocumentEmbedding.dimension` and `BaseVectorRetriever.dimension` reject vectors with mismatched shapes (`len(v) != 384`) with a clear `ValueError`.
  - Database level: PostgreSQL strictly enforces length constraints; inserting a vector of length $\ne 384$ raises a database exception.

### 13.2 Strategy for Coexisting Multi-Dimensional Models (768d, 1024d)
When future milestones introduce larger models (e.g. `google/muril-base-cased` at $d=768$ or `BAAI/bge-m3` at $d=1024$), multiple model dimensions must coexist. LinguaLens designs for two architectural evolution paths:
1. **Unconstrained Vectors (Native pgvector 0.5.0+):**
   - By declaring `embedding vector` (omitting the dimension parameter), PostgreSQL allows vectors of arbitrary dimensions in the same table.
   - However, an HNSW index in pgvector requires a fixed dimension per index. Separate partial or expression indexes must be defined per dimension:
     ```sql
     CREATE INDEX idx_embeddings_384 ON document_embeddings USING hnsw (embedding vector_cosine_ops) WHERE dimension = 384;
     CREATE INDEX idx_embeddings_768 ON document_embeddings USING hnsw (embedding vector_cosine_ops) WHERE dimension = 768;
     CREATE INDEX idx_embeddings_1024 ON document_embeddings USING hnsw (embedding vector_cosine_ops) WHERE dimension = 1024;
     ```
2. **Table Partitioning / Dedicated Embedding Tables:**
   - Partitioning `document_embeddings` by `model_id` or dimension into dedicated physical tables (`document_embeddings_384`, `document_embeddings_768`, etc.).
   - This provides optimal index density, distinct memory cache pools, and clean migration paths per model family.


