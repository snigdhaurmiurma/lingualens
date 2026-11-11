# LinguaLens — 60–90 Second Video Demo Storyboard
*Portfolio & LinkedIn Video Walkthrough Script*

---

### **Video Overview**
* **Target Duration:** 75–90 seconds
* **Target Audience:** ML Engineers, Search/IR Practitioners, NLP Researchers, Hiring Managers
* **Audio / Style:** Clear, professional, research-focused voiceover or crisp text captions with ambient lo-fi synth background audio.
* **Recording Resolution:** 1080p (1920x1080) at 60 FPS

---

### **Scene-by-Scene Storyboard**

#### **Scene 1: The Hook & Research Motivation (0:00 – 0:18)**
* **Visual:** Smooth pan over the LinguaLens Overview Dashboard (`https://lingualens-fhpq.vercel.app`). Highlight the live **`Gateway Connected`** indicator, the **Model Adapter Registry** (MiniLM vs Multilingual-E5), and the active **5-Language Matrix**.
* **Voiceover / Caption:**
  > *"When we evaluate multilingual semantic search, we often assume high-resource language benchmarks translate equally to global languages. But what actually happens when users search in Bengali, Hindi, Arabic, or phonetic transliterations like Banglish? Meet LinguaLens — an open-source evaluation platform designed to quantify cross-lingual retrieval disparity and subword tokenization bias."*

#### **Scene 2: Multilingual Query Lab in Action (0:18 – 0:45)**
* **Visual:** Navigate to `/query-lab`.
  1. Select `paraphrase-multilingual-MiniLM-L12-v2`.
  2. Select Bengali (`bn`) and load the suggestion: `"কৃষি ক্ষেত্রে জলবায়ু পরিবর্তনের প্রভাব"`.
  3. Expand the **Tokenizer Inspector**: Show subword tokens and the **subword fertility ratio** (~2.67x).
  4. Click **Execute Semantic Search**: Show ranked candidate cards, cosine similarity scores, and execution latency.
  5. Switch query language to Banglish (`bn-Latn`): `"krishi khetre jolobayu poribortoner probhab"`. Re-execute and observe score degradation.
* **Voiceover / Caption:**
  > *"In the Query Lab, we can interactively test dense vector retrieval across scripts. Notice how Bengali generates 2.67 subwords per word due to vocabulary fragmentation. When we switch from native Bengali script to phonetic Banglish, semantic similarity drops significantly — revealing the severe Transliteration Script Penalty of modern embedding models."*

#### **Scene 3: Dual-Mode Architecture & Tokenizer Inspection (0:45 – 1:05)**
* **Visual:** Toggle between `MiniLM-L12-v2` and `multilingual-e5-small`. Highlight automatic `"query: "` / `"passage: "` prefix injection for E5. Switch retrieval mode between `EXACT` (cosine brute-force) and `VECTOR_INDEX` (HNSW).
* **Voiceover / Caption:**
  > *"LinguaLens features a clean adapter registry supporting both MiniLM and Multilingual-E5 with strict prompt prefix handling. The retrieval layer supports dual-mode search: exact cosine similarity for ground-truth benchmarking, and pgvector HNSW indexing for scalable production simulation."*

#### **Scene 4: Statistical Evaluation & Fairness Matrix (1:05 – 1:20)**
* **Visual:** Navigate to `/experiments/` and show an evaluated study summary:
  - Metric Table: Precision@10, Recall@10, MRR, nDCG@10 with **95% bootstrap confidence intervals**.
  - Fairness Disparity Table: Relative Disparity Ratio ($RDR$) and Paired Wilcoxon $p$-values ($p < 0.05^*$).
* **Voiceover / Caption:**
  > *"Every evaluation is statistically rigorous. LinguaLens computes non-parametric bootstrap confidence intervals and paired significance tests, proving whether retrieval performance differences between languages represent true systemic bias or random sampling variance."*

#### **Scene 5: Architecture & Open Source Call-to-Action (1:20 – 1:30)**
* **Visual:** Show clean terminal running Next.js 14 and FastAPI + GitHub repository link (`https://github.com/snigdhaurmiurma/lingualens`) and live Vercel link (`https://lingualens-fhpq.vercel.app`).
* **Voiceover / Caption:**
  > *"Built with Next.js 14, FastAPI, PyTorch, and SQLAlchemy. Full architecture documentation, reproducible test suites, and source code are available on GitHub, and the live platform is deployed on Vercel. Check out the link below!"*
