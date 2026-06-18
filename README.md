# AutoRAG — Self-Reflective Retrieval-Augmented Generation

Açık erişimli akademik makaleler üzerinde çalışan, retrieval kalitesini kendi değerlendiren ve gerekirse sorguyu yeniden yazarak tekrar retrieval yapan **Self-Reflective Auto-RAG** sistemi.

---

## Proje Amacı

Geleneksel RAG sistemleri, bir kez retrieval yaptıktan sonra elde ettikleri bağlamla cevap üretir. Bu sistemde:

1. Kullanıcı sorgusu ArXiv ve CORE API üzerinden ilgili makaleler getirilerek yanıtlanmaya çalışılır.
2. Retrieval sonucu bir **grade node** tarafından LLM ile değerlendirilir (relevant / not relevant).
3. Sonuç yetersizse sorgu **rewrite node** tarafından yeniden yazılır ve retrieval tekrarlanır.
4. Yeterli bağlam bulununca **generate node** nihai cevabı ve atıfları üretir.
5. Tüm süreç [RAGAS](https://github.com/explodinggradients/ragas) metrikleriyle değerlendirilir.

---

## Sistem Mimarisi

```
Kullanıcı Sorgusu
       │
       ▼
┌─────────────────┐
│  Data Fetching  │  arxiv_fetcher + core_fetcher → PaperMetadata
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunking     │  RecursiveCharacterTextSplitter → Chunk[]
└────────┬────────┘
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
│Generate│  │Rewrite Node │ → tekrar Retrieve (max 3 iterasyon)
│  Node  │  └─────────────┘
└────┬───┘
     │
     ▼
Cevap + Atıflar + RAGAS Metrikleri
```

**Modüller:**

| Klasör | Sorumluluk |
|--------|------------|
| `data/` | ArXiv & CORE API'den makale çekme, PDF indirme |
| `vectordb/` | ChromaDB ve FAISS vektör depo adaptörleri |
| `retrieval/` | BM25 sparse retrieval, RRF fusion |
| `graph/` | LangGraph state, grade node, generate node |
| `eval/` | RAGAS değerlendirme, dense vs hybrid karşılaştırma |
| `ui/` | Streamlit arayüzü (sorgu, iz takibi, dashboard) |
| `tests/` | pytest tabanlı birim ve entegrasyon testleri |

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

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

### Ortam Değişkenleri

```bash
cp .env.example .env
```

`.env` dosyasını düzenle:

```
API_KEY_ARXIV=your_arxiv_api_key      # Opsiyonel — anonim erişim çalışır
CORE_API_KEY=your_core_api_key        # https://core.ac.uk/services/api
OPENAI_API_KEY=your_openai_api_key    # Grade ve Generate node'ları için
```

---

## Test Çalıştırma

```bash
python -m pytest
```

Belirli bir modülü test etmek için:

```bash
python -m pytest tests/test_grade_node.py -v
python -m pytest tests/test_bm25_retriever.py -v
```

Tüm testler mock kullanır; API anahtarı gerekmez.

---

## Streamlit UI Çalıştırma

```bash
streamlit run ui/app.py
```

Arayüz beş sayfadan oluşur (sol sidebar'dan seçilir):

| Sayfa | İçerik |
|-------|--------|
| **RAG Sorgusu** | Sorgu, retrieval izi (chunk → grade → rewrite), cevap + atıflar |
| **Metrics Dashboard** | `results/*.json` dosyalarından RAGAS metrikleri, latency, rewrite grafikleri |
| **Model Comparison** | Model bazlı faithfulness / relevancy / maliyet karşılaştırması |
| **Benchmark Results** | Standard vs Auto-RAG · Retriever · Chunk Size · Top-k sekmeleri |
| **Retrieved Sources** | Son sorgudaki chunk'lar, skorlar, makale linkleri ve atıf listesi |

---

## Deneyleri Çalıştırma

### Veri Seti

`data/test_queries.json` — 25 akademik Q&A sorusu (RAG, BM25, Hybrid, Faithfulness vb. konuları kapsar). Tüm `--dataset` / `--queries` parametreleri bu dosyayı gösterir.

### RAGAS Değerlendirme (Standard RAG vs Auto-RAG)

```bash
python -m eval.eval_runner --dataset data/test_queries.json --output results/eval.json
```

### Retrieval Karşılaştırması

```bash
python -m eval.retrieval_experiment --retrieval-mode dense   --queries data/test_queries.json
python -m eval.retrieval_experiment --retrieval-mode hybrid  --queries data/test_queries.json
```

### Chunk Size Karşılaştırması

```bash
python -m eval.chunking_experiment --chunk-size 512  --dataset data/test_queries.json
python -m eval.chunking_experiment --chunk-size 1024 --dataset data/test_queries.json
```

### Top-k Karşılaştırması

```bash
python -m eval.topk_experiment --dataset data/test_queries.json
```

### Model Karşılaştırması

```bash
python -m eval.model_comparison --dataset data/test_queries.json --models gpt-4o-mini gpt-4o
```

### Tüm Benchmark Suite

```bash
python -m eval.benchmark_runner --dataset data/test_queries.json --benchmark all
```

---

## Metrikler

| Metrik | Açıklama |
|--------|----------|
| **Faithfulness** | Üretilen cevabın, retrieval edilen bağlamla ne kadar tutarlı olduğu (hallüsinasyon tespiti) |
| **Answer Relevancy** | Cevabın kullanıcı sorusuna ne kadar ilgili olduğu |
| **Context Precision** | Retrieval edilen chunk'ların soruyla ilgili olanların oranı |
| **Context Recall** | Cevap için gereken bilgilerin retrieval ile ne kadarının bulunduğu |

Tüm metrikler 0–1 aralığında; 1 en iyi değerdir.

---

## Sonuç Dosyaları

Tüm deney çıktıları `results/` klasörüne zaman damgalı JSON olarak yazılır:

```
results/
├── benchmark_rag_comparison.json   # Standard RAG vs Auto-RAG
├── benchmark_retriever.json        # bm25 / dense / hybrid karşılaştırması
├── benchmark_chunk.json            # chunk size 256/512/1024/2048
├── benchmark_topk.json             # top-k 3/5/10/15
└── benchmark_models.json           # model karşılaştırması
```

Her JSON dosyası `experiments` listesi içerir; her satır aşağıdaki alanları taşır:

```json
{
  "experiment_name": "retriever_hybrid",
  "experiment_type": "retrieval",
  "timestamp": "2025-01-01T12:00:00",
  "model_name": null,
  "retriever_type": "hybrid",
  "chunk_size": null,
  "top_k": 5,
  "faithfulness": 0.85,
  "answer_relevancy": 0.88,
  "context_precision": 0.76,
  "context_recall": 0.71,
  "avg_latency_seconds": 0.12,
  "avg_rewrite_count": 0.0,
  "n_questions": 25
}
```

---

## Katkıda Bulunanlar

- Burcu Yıldırım — [@burcuyldrm](https://github.com/burcuyldrm)
- Merve Çaloğlu — [@mervecaloglu](https://github.com/mervecaloglu)
