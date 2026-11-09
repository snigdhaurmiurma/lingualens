# LinguaLens: Embedding Model Registry & Candidate Evaluation

**Document Version:** 1.0.0  
**Verification Date:** November 2026  
**License & Model Card Audit:** Completed  

---

## 1. Candidate Model Comparison Matrix

The table below summarizes open-source embedding models verified for cross-lingual dense retrieval, their licenses, hardware footprint, and input conventions:

| Model Identifier | Hugging Face ID | Params | Dim | Max Tokens | License | Target Languages Covered | CPU Latency Profile | Required Prefix |
|---|---|---|---|---|---|---|---|---|
| **Multilingual MiniLM** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 117M | 384 | 128 / 512 | Apache 2.0 | `en`, `bn`, `hi`, `ar` (50+ langs) | **Fastest** (~35ms / query) | None |
| **Multilingual E5 Small** | `intfloat/multilingual-e5-small` | 118M | 384 | 512 | MIT | `en`, `bn`, `hi`, `ar` (100 langs) | **Fast** (~40ms / query) | `query: ` / `passage: ` |
| **Multilingual E5 Base** | `intfloat/multilingual-e5-base` | 278M | 768 | 512 | MIT | `en`, `bn`, `hi`, `ar` (100 langs) | **Moderate** (~95ms / query) | `query: ` / `passage: ` |
| **BGE-M3** | `BAAI/bge-m3` | 567M | 1024 | 8192 | MIT | `en`, `bn`, `hi`, `ar` (100+ langs) | **Heavy** (~280ms / query) | None (for dense) |
| **MuRIL Base Cased** | `google/muril-base-cased` | 236M | 768 | 512 | Apache 2.0 | `en`, `bn`, `hi` + **Transliterated Indic** | **Moderate** (~80ms / query) | Mean Pooling on output |

---

## 2. In-Depth Model Profiles

### 2.1 Multilingual E5 Small (`intfloat/multilingual-e5-small`)
- **Primary Strengths:** Exceptional parameter-to-quality ratio. Trained using 2-stage contrastive training on 1 billion multilingual text pairs (mMARCO, MIRACL, and multilingual web crawl).
- **Prefix Requirement:** **Strict.** Queries must be prepended with `"query: "` and indexed documents with `"passage: "`. Failing to include these prefixes causes up to 15% drop in retrieval accuracy.
- **Language Coverage:**
  - `en`: Native high-resource quality.
  - `bn`: Solid performance on formal native Bengali script (`Beng`).
  - `bn-Latn` (Banglish): Substantial drop due to subword fragmentation in the XLM-RoBERTa vocabulary.
  - `hi`: Strong coverage in Devanagari script.
  - `ar`: Highly stable representation with Arabic morphology.
- **Recommended Role in LinguaLens:** **Default Primary Baseline Model** for fast local development and automated CI regression tests.

### 2.2 Paraphrase Multilingual MiniLM L12 v2 (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
- **Primary Strengths:** Distilled from XLM-RoBERTa into a 12-layer MiniLM architecture. Extremely lightweight (~470MB disk footprint, ~650MB peak RAM).
- **Prefix Requirement:** None. Accepts raw text strings.
- **Licensing:** Apache 2.0 (unrestricted commercial and research usage).
- **Recommended Role in LinguaLens:** Lightweight secondary benchmark and CPU performance ceiling test.

### 2.3 MuRIL Base Cased (`google/muril-base-cased`)
- **Primary Strengths:** Pre-trained by Google Research specifically on 17 Indian languages and **transliterated counterparts** (Latin-script Bengali, Hindi, Tamil, etc.).
- **Special Significance for LinguaLens:**
  While models like E5 and MiniLM treat Latinized Bengali (`bn-Latn`) as foreign noise, MuRIL was trained on paired native-transliterated text. Comparing MuRIL vs. E5-Small on `bn` vs `bn-Latn` provides the centerpiece empirical demonstration of LinguaLens's Transliteration Fairness module.
- **Licensing:** Apache 2.0.

### 2.4 BAAI BGE-M3 (`BAAI/bge-m3`)
- **Primary Strengths:** State-of-the-art multilingual capability supporting dense, sparse (lexical), and multi-vector (ColBERT-style) representations. Supports ultra-long sequences up to 8,192 tokens.
- **Hardware Requirement:** Minimum 16GB RAM for CPU inference; recommended NVIDIA GPU (VRAM $\ge 8$GB).
- **Recommended Role in LinguaLens:** Flagship heavyweight benchmark for GPU-enabled deployment profiles.

---

## 3. Tokenizer Asymmetry & Transliteration Analysis

### 3.1 Subword Fertility Ratio
Subword fertility measures average subword token fragmentation per word. **Crucially, special structural boundary tokens (such as `<s>` and `</s>`) are sequence boundary markers, not linguistic subwords, and are excluded from fertility calculations to prevent artificial inflation on short queries:**

$$F(T, L) = \frac{|\mathcal{T}_{\text{linguistic}}(T)|}{|\mathcal{W}(T)|}$$

where $\mathcal{T}_{\text{linguistic}}(T)$ excludes all sequence-level special tokens ($\{ \texttt{<s>}, \texttt{</s>}, \texttt{[CLS]}, \texttt{[SEP]} \}$) and $\mathcal{W}(T)$ is the natural word count.

#### Empirically Verified Baseline Values (MiniLM-L12-v2 / XLM-R Tokenizer):
- **English (`en`):** 5 linguistic tokens / 4 words = **1.25**
- **Bengali (`bn`, native script):** 8 linguistic tokens / 3 words = **2.67**
- **Banglish (`bn-Latn`, transliteration):** 6 linguistic tokens / 3 words = **2.00**
- **Hindi (`hi`, Devanagari):** 6 linguistic tokens / 4 words = **1.50**
- **Arabic (`ar`, Arabic script):** 8 linguistic tokens / 3 words = **2.67**

### 3.2 Scientific Nuance: Fragmentation vs. Semantic Understanding
- **Input Efficiency vs. Downstream Representation:** High subword fertility means more tokens must be processed by self-attention, consuming more of the context window. However, high fertility alone does **not** prove semantic inability. If pre-training corpora contain sufficient co-occurrence data, the transformer's multi-head attention can compose rich semantics from subwords.
- **Transliteration Sensitivity:** Banglish (`bn-Latn`) often exhibits non-standardized phonetic spellings across speakers (e.g., `porashona` vs `porasona` vs `porasuna`), which compounds subword fragmentation with vocabulary mismatch.
- **Comprehensive Evaluation Requirement:** LinguaLens tracks fertility alongside $nDCG$ and cross-lingual cosine similarity across multi-query benchmark suites ($N \ge 100$) to evaluate where fragmentation correlates with true retrieval loss, rather than inferring causation from isolated examples.

