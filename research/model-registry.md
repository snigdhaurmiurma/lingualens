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
The subword fertility ratio $F(T, L)$ is defined as:

$$F(T, L) = \frac{\text{Total Subword Tokens Produced by Tokenizer } T}{\text{Total Whitespace-Delimited Words in Text of Language } L}$$

In typical SentencePiece/Byte-Pair Encoding models (like XLM-RoBERTa):
- English (`en`): $F \approx 1.1 - 1.3$
- Bengali (`bn`, native script): $F \approx 1.6 - 2.2$
- Banglish (`bn-Latn`, transliteration): $F \approx 3.2 - 5.1$

### 3.2 Consequences of High Fertility
When a single Banglish word (e.g., `porashona` - "studies") is fragmented into 4 subwords (`["por", "##ash", "##on", "##a"]`), the attention mechanism dissipates over fragmented n-grams, leading to severe vector drift and retrieval collapse. LinguaLens explicitly tracks and plots fertility ratios alongside $nDCG$ to diagnose this root cause.
