# AutoRAG — Self-Reflective Retrieval-Augmented Generation

**Self-Reflective Auto-RAG: Açık Erişimli Akademik Makaleler Üzerinde Otonom Hata Denetimi ve Parametrik Performans Analizi**

Açık erişimli akademik makaleler üzerinde çalışan, retrieval kalitesini kendi değerlendiren, gerekirse sorguyu yeniden yazarak tekrar retrieval yapan ve üretilen cevabın bağlamla tutarlılığını (faithfulness) denetleyen **Self-Reflective Auto-RAG** sistemi.

---

## Project Overview

Geleneksel RAG sistemleri, bir kez retrieval yaptıktan sonra elde ettikleri bağlamla cevap üretir. Bu projede:

1. Kullanıcı sorgusu ArXiv ve CORE API üzerinden ilgili makaleler getirilerek yanıtlanmaya çalışılır.
2. Retrieval sonucu bir **grade node** tarafından LLM ile değerlendirilir (relevant / not relevant).
3. Sonuç yetersizse sorgu **rewrite node** tarafından yeniden yazılır ve retrieval tekrarlanır (self-reflection loop).
4. Yeterli bağlam bulununca **generate node** nihai cevabı ve atıfları üretir.
5. **Faithfulness node** üretilen cevabın retrieval edilen bağlamla desteklenip desteklenmediğini denetler.
6. Tüm süreç [RAGAS](https://github.com/explodinggradients/ragas) metrikleriyle değerlendirilir.

---

## Architecture

```
Kullanıcı Sorgusu
       │
       ▼
┌─────────────────────────────┐
│  Hybrid Retrieval           │
│  Dense (FAISS/ChromaDB)     │
│  + Sparse (BM25)  → RRF    │  Reciprocal Rank Fusion
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│   Grade Node    │  LLM → GradeResult {relevant, confidence, reasoning}
└────────┬────────┘
         │
    relevant?
    ┌─────┴──────┐
   YES           NO
    │             │
    ▼             ▼
┌────────┐  ┌─────────────┐
│Generate│  │Rewrite Node │ → tekrar Retrieve (max 2 iterasyon)
│  Node  │  └─────────────┘
└────┬───┘
     │
     ▼
┌──────────────────┐
│ Faithfulness Node│  LLM veya heuristic → {faithful, confidence, unsupported_claims}
└────────┬─────────┘
         │
         ▼
Cevap + Atıflar + RAGAS Metrikleri + Faithfulness
```

**Modüller:**

| Klasör | Sorumluluk |
|--------|------------|
| `data/` | ArXiv & CORE API'den makale çekme, PDF indirme, `qa_dataset.json` |
| `vectordb/` | ChromaDB ve FAISS vektör depo adaptörleri |
| `retrieval/` | BM25 sparse retrieval, RRF fusion |
| `graph/` | LangGraph state, grade, generate, rewrite, faithfulness node'ları |
| `eval/` | RAGAS değerlendirme, dataset loader, benchmark runner |
| `paper_assets/` | Methodology diagram (Mermaid) |
| `ui/` | Streamlit arayüzü (sorgu, iz takibi, dashboard, CSV download) |
| `tests/` | pytest tabanlı birim ve entegrasyon testleri |

---

## Dataset Format

`data/qa_dataset.json` — 30 akademik Q&A sorusu. Her kayıt aşağıdaki alanları taşır:

```json
{
  "id": "q001",
  "question": "What is Retrieval-Augmented Generation (RAG)?",
  "ground_truth": "RAG is a technique that combines retrieval ...",
  "expected_source": "RAG paper (Lewis et al., 2020)",
  "topic": "RAG",
  "difficulty": "easy"
}
```

Konular: RAG, dense retrieval, BM25, hybrid retrieval, query rewriting, faithfulness, context precision, context recall, vector database, chunking, top-k retrieval.

Dataset yüklemek ve doğrulamak için:

```python
from eval.dataset_loader import load_qa_dataset, validate_qa_dataset

samples = load_qa_dataset("data/qa_dataset.json")
validate_qa_dataset(samples)   # raises ValueError on issues
```

---

## Kurulum

### Gereksinimler

- **Python 3.11** veya **3.12** (önerilen)
- pip

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/burcuyldrm/autoRAG.git
cd autoRAG

# 2. Sanal ortam oluştur
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate   # Windows

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

### Ortam Değişkenleri

```bash
cp .env.example .env
```

`.env` dosyasını düzenle:

```
OPENAI_API_KEY=your_openai_api_key    # Grade, Generate, Faithfulness node'ları için
CORE_API_KEY=your_core_api_key        # https://core.ac.uk/services/api
```

---

## Running Standard vs Auto-RAG Benchmark

Standard RAG (Question → Retriever → Top-k chunks → Generate → Answer + sources) ile
Self-Reflective Auto-RAG (Question → Retrieve → Grade → Rewrite if needed → Generate → Faithfulness → Answer + sources + metrics) karşılaştırması:

```bash
python -m eval.benchmark_runner --dataset data/qa_dataset.json --benchmark rag
```

Çıktı: `results/benchmark_rag_comparison.json`

```json
{
  "benchmark_name": "standard_vs_autorag",
  "dataset_path": "data/qa_dataset.json",
  "n_questions": 30,
  "experiments": [
    {
      "experiment_name": "standard_rag",
      "experiment_type": "comparison",
      "faithfulness": 0.0,
      "answer_relevancy": 0.0,
      "context_precision": 0.0,
      "context_recall": 0.0,
      "avg_latency_seconds": 0.05,
      "avg_rewrite_count": 0.0,
      "n_questions": 30
    },
    {
      "experiment_name": "auto_rag",
      "avg_rewrite_count": 1.0,
      ...
    }
  ]
}
```

---

## Running Retriever Comparison

BM25, dense ve hybrid retriever karşılaştırması:

```bash
python -m eval.benchmark_runner --dataset data/qa_dataset.json --benchmark retriever
```

Çıktı: `results/benchmark_retriever.json`

```json
{
  "benchmark_name": "retriever_comparison",
  "dataset_path": "data/qa_dataset.json",
  "n_questions": 30,
  "experiments": [
    {"experiment_name": "retriever_bm25", "experiment_type": "retrieval", "retriever_type": "bm25", ...},
    {"experiment_name": "retriever_dense", "retriever_type": "dense", ...},
    {"experiment_name": "retriever_hybrid", "retriever_type": "hybrid", ...}
  ]
}
```

---

## Faithfulness Evaluation

`graph/nodes/faithfulness_node.py` üretilen cevabın retrieval edilen bağlamla desteklenip desteklenmediğini kontrol eder.

**LLM varken:** LLM her iddiayı bağlamda arar ve desteklenmeyen iddiaları listeler.

**LLM yoksa (fallback heuristic):** Cevap ve bağlam arasındaki token örtüşme oranına göre güven skoru hesaplanır.

```python
from graph.nodes.faithfulness_node import check_faithfulness

result = check_faithfulness(
    question="What is BM25?",
    answer="BM25 is a ranking function based on term frequency.",
    contexts=["BM25 (Best Match 25) is a probabilistic ranking function ..."],
    llm=None,   # None → fallback heuristic
)
# result: {"faithful": True, "confidence": 0.72, "unsupported_claims": [], "reasoning": "..."}
```

Her `AutoRAGChain.run()` ve `StandardRAGChain.run()` çağrısı otomatik olarak `faithfulness_result`, `faithfulness_score`, `unsupported_claims` alanlarını döndürür.

---

## Results JSON Format

Tüm deney çıktıları `results/` klasörüne JSON olarak yazılır. Her dosya `experiments` listesi içerir:

```json
{
  "benchmark_name": "...",
  "dataset_path": "...",
  "n_questions": 30,
  "timestamp": "2026-06-20T10:00:00",
  "experiments": [
    {
      "experiment_name": "...",
      "experiment_type": "...",
      "timestamp": "...",
      "model_name": null,
      "retriever_type": null,
      "chunk_size": null,
      "top_k": null,
      "faithfulness": null,
      "answer_relevancy": null,
      "context_precision": null,
      "context_recall": null,
      "avg_latency_seconds": 0.05,
      "avg_rewrite_count": 0.0,
      "n_questions": 30
    }
  ]
}
```

---

## Streamlit Dashboard

```bash
streamlit run ui/app.py
```

| Sayfa | İçerik |
|-------|--------|
| **RAG Sorgusu** | Sorgu, retrieval izi (chunk → grade → rewrite → faithfulness), cevap + atıflar |
| **Metrics Dashboard** | `results/*.json` dosyalarından RAGAS metrikleri, faithfulness, latency grafikleri + CSV download |
| **Model Comparison** | Model bazlı faithfulness / relevancy / maliyet karşılaştırması |
| **Benchmark Results** | Standard vs Auto-RAG · Retriever · Chunk Size · Top-k sekmeleri + CSV download |
| **Retrieved Sources** | Son sorgudaki chunk'lar, skorlar, makale linkleri ve atıf listesi |

Her iki benchmark sayfası (Metrics Dashboard ve Benchmark Results) CSV indirme butonu içerir.

---

## Methodology Diagram

Mermaid formatında diyagram: `paper_assets/methodology_diagram.mmd`

PNG çıktısı üretmek için (mmdc CLI gereklidir):

```bash
npx -p @mermaid-js/mermaid-cli mmdc \
  -i paper_assets/methodology_diagram.mmd \
  -o paper_assets/methodology_diagram.png
```

Diyagram iki akışı gösterir:
- **A) Standard RAG:** Question → Retriever → Top-k Contexts → Generator → Answer + Sources
- **B) Self-Reflective Auto-RAG:** Question → Retriever → Retrieved Contexts → Relevance Grader → (düşük relevance → Query Rewriter → Retriever; yeterli → Generator → Faithfulness Checker → Answer + Sources + Metrics)

---

## Deneyleri Çalıştırma / Reproducing Experiments

### Tüm Benchmark Suite

```bash
python -m eval.benchmark_runner --dataset data/qa_dataset.json --benchmark all
```

### Bireysel Benchmark'lar

```bash
# Standard RAG vs Auto-RAG
python -m eval.benchmark_runner --dataset data/qa_dataset.json --benchmark rag

# Retriever karşılaştırması (BM25 / Dense / Hybrid)
python -m eval.benchmark_runner --dataset data/qa_dataset.json --benchmark retriever

# Chunk size karşılaştırması (256 / 512 / 1024 / 2048)
python -m eval.benchmark_runner --dataset data/qa_dataset.json --benchmark chunk
```

### RAGAS Değerlendirme (Standard RAG vs Auto-RAG)

```bash
python -m eval.eval_runner --dataset data/qa_dataset.json --output results/eval.json
```

### Top-k ve Model Karşılaştırması

```bash
python -m eval.topk_experiment --dataset data/qa_dataset.json
python -m eval.model_comparison --dataset data/qa_dataset.json --models gpt-4o-mini gpt-4o
```

### Test Çalıştırma

```bash
python -m pytest
```

Belirli bir modül:

```bash
python -m pytest tests/test_faithfulness_node.py -v
python -m pytest tests/test_dataset_loader.py -v
python -m pytest tests/test_benchmark_rag_comparison.py -v
python -m pytest tests/test_retriever_comparison_benchmark.py -v
```

---

## Metrikler

| Metrik | Açıklama |
|--------|----------|
| **Faithfulness** | Üretilen cevabın, retrieval edilen bağlamla ne kadar tutarlı olduğu (hallüsinasyon tespiti) |
| **Answer Relevancy** | Cevabın kullanıcı sorusuna ne kadar ilgili olduğu |
| **Context Precision** | Retrieval edilen chunk'ların soruyla ilgili olanların oranı |
| **Context Recall** | Cevap için gereken bilgilerin retrieval ile ne kadarının bulunduğu |
| **Faithfulness Score** | Heuristic veya LLM tabanlı faithfulness güven skoru (0–1) |

---

## Katkıda Bulunanlar

- Burcu Yıldırım — [@burcuyldrm](https://github.com/burcuyldrm)
- Merve Çaloğlu — [@mervecaloglu](https://github.com/mervecaloglu)
