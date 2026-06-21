#!/usr/bin/env python3
"""
Auto-RAG Projesi — Tüm Issues'ları otomatik oluştur
Gerekli: pip install PyGithub
"""

from github import Github
import os
from datetime import datetime

# GitHub Token'ı ortam değişkeninden al
# Token'ı şu şekilde ayarla: export GITHUB_TOKEN='ghp_xxxx'
TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    print("❌ GITHUB_TOKEN çevre değişkeni ayarlanmamış!")
    print("   Linux/Mac: export GITHUB_TOKEN='ghp_xxxx'")
    print("   Windows: set GITHUB_TOKEN=ghp_xxxx")
    exit(1)

# GitHub instance
g = Github(TOKEN)
repo = g.get_repo("burcuyldrm/autoRAG")

# Tüm görevler
TASKS = [
    {
        "number": "T01",
        "title": "Project Scaffolding & Environment Setup",
        "assignee": "mervecalogluuu",
        "labels": ["devops", "documentation"],
        "description": """## 📋 Açıklama
Auto-RAG projesinin temel yapılandırmasını ve ortamını hazırla.

## 📝 Yapılacaklar
- [ ] Python monorepo yapısı oluşturulmuş
- [ ] pyproject.toml veya requirements.txt hazırlı
- [ ] Klasörleri oluşturulmuş: /data, /vectordb, /graph, /eval, /ui
- [ ] .env şablonu oluşturulmuş (API_KEY_ARXIV, CORE_API_KEY, etc)
- [ ] .gitignore dosyası hazırlı
- [ ] README.md başlangıç dosyası yazılmış

## ✨ Kabul Kriterleri
- [x] Monorepo yapısı kurulu
- [x] requirements.txt yazılmış
- [x] .env.example hazırlı
- [x] Test çalışabiliyor

## 🌿 Branch
feature/project-setup

## 📦 Bağımlılıklar
Hiç""",
    },
    {
        "number": "T02",
        "title": "ArXiv API Entegrasyonu",
        "assignee": "mervecalogluuu",
        "labels": ["backend", "data"],
        "description": """## 📋 Açıklama
ArXiv Python kütüphanesi kullanarak makale sorgula, metadata ve PDF URL'lerini al.
PDFs'i /data/raw klasörüne indir. Rate limiting ve retry logic ekle.

## 📝 Yapılacaklar
- [ ] arxiv_fetcher.py dosyası oluşturulmuş
- [ ] ArXiv'den keyword ile makale sorgulanabiliyor
- [ ] Metadata: başlık, abstract, arXiv ID, PDF URL döndürülüyor
- [ ] Rate limiting mekanizması (max 3 istek/saniye)
- [ ] Retry logic (max 3 deneme, 2 saniye bekleme)
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] ArXiv fetcher çalışıyor
- [x] Rate limiting aktif
- [x] Testler pass ediyor

## 🌿 Branch
feature/arxiv-fetcher

## 📦 Bağımlılıklar
T01

## 🔗 Engel Kaldırır
T04""",
    },
    {
        "number": "T03",
        "title": "CORE API Entegrasyonu",
        "assignee": "burcuyldrm",
        "labels": ["backend", "data"],
        "description": """## 📋 Açıklama
CORE API kullanarak açık erişim makalelerini çek. ArXiv fetcher'ı ile birleştir.
Unified PaperFetcher interface'i oluştur.

## 📝 Yapılacaklar
- [ ] core_fetcher.py dosyası oluşturulmuş
- [ ] CORE API'den makale sorgulanabiliyor
- [ ] PaperFetcher abstract class'ı oluşturulmuş
- [ ] ArXivFetcher ve CoreFetcher, PaperFetcher'dan miras alıyor
- [ ] Birleştirilmiş sonuçlar döndürülüyor
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] CORE fetcher çalışıyor
- [x] PaperFetcher interface tanımlandı
- [x] Testler pass ediyor

## 🌿 Branch
feature/core-fetcher

## 📦 Bağımlılıklar
T01, T02

## 🔗 Engel Kaldırır
T04""",
    },
    {
        "number": "T04",
        "title": "PDF Metin Çıkarma Pipeline",
        "assignee": "mervecalogluuu",
        "labels": ["backend", "data"],
        "description": """## 📋 Açıklama
PyMuPDF (fitz) kullanarak PDF'lerden metin çıkar. Sayfa numaraları korunmalı.
Çok sütunlu layoutları işle.

## 📝 Yapılacaklar
- [ ] pdf_extractor.py dosyası oluşturulmuş
- [ ] Metin sayfa numarası ile çıkarılıyor
- [ ] JSON yapısı: {paper_id, page, text}
- [ ] Çok sütunlu PDF'ler işleniyor
- [ ] Hata yönetimi (bozuk PDF, encrypted PDF)
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] PDF extraction çalışıyor
- [x] Sayfa numaraları korunuyor
- [x] Testler pass ediyor

## 🌿 Branch
feature/pdf-extractor

## 📦 Bağımlılıklar
T02, T03

## 🔗 Engel Kaldırır
T05, T06""",
    },
    {
        "number": "T05",
        "title": "512-Char Chunking Stratejisi",
        "assignee": "mervecalogluuu",
        "labels": ["backend", "data"],
        "description": """## 📋 Açıklama
RecursiveCharacterTextSplitter kullanarak metni 512 karakterlik parçalara böl.
50 karakter overlap ekle. Metadata ile birlikte Chunk nesneleri oluştur.

## 📝 Yapılacaklar
- [ ] chunker.py dosyası oluşturulmuş
- [ ] ChunkerConfig sınıfı oluşturulmuş (CHUNK_SIZE, OVERLAP)
- [ ] Chunk TypedDict'i oluşturulmuş (id, text, metadata)
- [ ] 512-char chunking çalışıyor
- [ ] Metadata korunuyor (paper_id, page, source_url)
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] 512-char chunking çalışıyor
- [x] Metadata korunuyor
- [x] Testler pass ediyor

## 🌿 Branch
feature/chunking-512

## 📦 Bağımlılıklar
T04

## 🔗 Engel Kaldırır
T07, T09""",
    },
    {
        "number": "T06",
        "title": "1024-Char Chunking Stratejisi",
        "assignee": "mervecalogluuu",
        "labels": ["backend", "data"],
        "description": """## 📋 Açıklama
1024 karakterlik chunking modu ekle. 100 karakter overlap. Mode'u konfigüre edilebilir yap.

## 📝 Yapılacaklar
- [ ] Chunker1024 sınıfı oluşturulmuş
- [ ] 1024-char + 100 overlap çalışıyor
- [ ] Config dosyasında CHUNK_SIZE seçeneği
- [ ] Dynamic mode switching çalışıyor
- [ ] Unit test yazılmış (512 vs 1024 karşılaştırması)

## ✨ Kabul Kriterleri
- [x] 1024-char chunking çalışıyor
- [x] Config switching çalışıyor
- [x] Testler pass ediyor

## 🌿 Branch
feature/chunking-1024

## 📦 Bağımlılıklar
T05

## 🔗 Engel Kaldırır
T07, T09, T21""",
    },
    {
        "number": "T07",
        "title": "ChromaDB Kurulumu & Ingestion Pipeline",
        "assignee": "burcuyldrm",
        "labels": ["backend", "vectordb"],
        "description": """## 📋 Açıklama
Persistent ChromaDB collection oluştur. Chunks'ları sentence-transformers
(all-MiniLM-L6-v2) ile embed et. ingest(chunks, mode) fonksiyonu ekle.

## 📝 Yapılacaklar
- [ ] vectorstore.py dosyası oluşturulmuş
- [ ] ChromaDB collection oluşturuluyor (/vectordb/chroma.db)
- [ ] Embedding model: all-MiniLM-L6-v2
- [ ] ingest() fonksiyonu çalışıyor
- [ ] Metadata vektörlerle birlikte depolanıyor
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] ChromaDB kurulu
- [x] Embedding çalışıyor
- [x] Testler pass ediyor

## 🌿 Branch
feature/chromadb-setup

## 📦 Bağımlılıklar
T05, T06

## 🔗 Engel Kaldırır
T09, T11""",
    },
    {
        "number": "T08",
        "title": "FAISS Alternatif Backend",
        "assignee": "burcuyldrm",
        "labels": ["backend", "vectordb"],
        "description": """## 📋 Açıklama
FAISS backend oluştur. ChromaDB ve FAISS'i VectorStore abstract class'ının arkasında sakla.

## 📝 Yapılacaklar
- [ ] VectorStore abstract class oluşturulmuş
- [ ] FAISSVectorStore sınıfı oluşturulmuş
- [ ] ChromaVectorStore refactor edildi
- [ ] Config'de DB seçimi (chroma/faiss)
- [ ] Her iki backend aynı interface'i kullanıyor
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] VectorStore interface tanımlandı
- [x] FAISS backend çalışıyor
- [x] Testler pass ediyor

## 🌿 Branch
feature/faiss-backend

## 📦 Bağımlılıklar
T07

## 🔗 Engel Kaldırır
T09, T11""",
    },
    {
        "number": "T09",
        "title": "Dense Vector Retriever",
        "assignee": "mervecalogluuu",
        "labels": ["backend", "retrieval"],
        "description": """## 📋 Açıklama
VectorRetriever.retrieve(query, k=5) fonksiyonu yaz. Cosine similarity
kullanarak chunks döndür.

## 📝 Yapılacaklar
- [ ] retriever.py dosyası oluşturulmuş
- [ ] VectorRetriever sınıfı oluşturulmuş
- [ ] retrieve(query, k) çalışıyor
- [ ] Cosine similarity ile sıralama
- [ ] Top-k results döndürülüyor
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] Dense retriever çalışıyor
- [x] Sıralama doğru
- [x] Testler pass ediyor

## 🌿 Branch
feature/vector-retriever

## 📦 Bağımlılıklar
T07, T08

## 🔗 Engel Kaldırır
T11""",
    },
    {
        "number": "T10",
        "title": "BM25 Sparse Retriever",
        "assignee": "burcuyldrm",
        "labels": ["backend", "retrieval"],
        "description": """## 📋 Açıklama
rank_bm25 kullanarak BM25 retriever oluştur. Ranked Chunk nesneleri döndür.

## 📝 Yapılacaklar
- [ ] BM25Retriever sınıfı oluşturulmuş
- [ ] retrieve(query, k) çalışıyor
- [ ] BM25 sıralaması çalışıyor
- [ ] Top-k results döndürülüyor
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] BM25 retriever çalışıyor
- [x] Sıralama doğru
- [x] Testler pass ediyor

## 🌿 Branch
feature/bm25-retriever

## 📦 Bağımlılıklar
T05, T06

## 🔗 Engel Kaldırır
T11""",
    },
    {
        "number": "T11",
        "title": "Hybrid Retriever (RRF Fusion)",
        "assignee": "mervecalogluuu",
        "labels": ["backend", "retrieval"],
        "description": """## 📋 Açıklama
VectorRetriever + BM25Retriever'ı Reciprocal Rank Fusion (RRF) ile birleştir.

## 📝 Yapılacaklar
- [ ] HybridRetriever sınıfı oluşturulmuş
- [ ] RRF algoritması uygulanmış
- [ ] Config'de hybrid mode seçeneği
- [ ] Birleştirilmiş sıralama döndürülüyor
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] Hybrid retriever çalışıyor
- [x] RRF sıralaması doğru
- [x] Testler pass ediyor

## 🌿 Branch
feature/hybrid-retriever

## 📦 Bağımlılıklar
T09, T10

## 🔗 Engel Kaldırır
T13""",
    },
    {
        "number": "T12",
        "title": "LangGraph State Schema & Skeleton",
        "assignee": "mervecalogluuu",
        "labels": ["backend", "graph"],
        "description": """## 📋 Açıklama
GraphState TypedDict'i tanımla. Graph skeleton'unu oluştur:
retrieve → grade → rewrite/generate

## 📝 Yapılacaklar
- [ ] schemas.py dosyası oluşturulmuş
- [ ] GraphState TypedDict tanımlandı
- [ ] graph_skeleton.py oluşturulmuş
- [ ] StateGraph oluşturulmuş (retrieve, grade, rewrite, generate node'ları)
- [ ] Initial state tanımlandı

## ✨ Kabul Kriterleri
- [x] Schemas tanımlandı
- [x] Graph skeleton hazırlandı
- [x] Node placeholder'ları var

## 🌿 Branch
feature/langgraph-schema

## 📦 Bağımlılıklar
T06, T08, T11

## 🔗 Engel Kaldırır
T13, T14, T15, T16, T17""",
    },
    {
        "number": "T13",
        "title": "Retrieve Node",
        "assignee": "mervecalogluuu",
        "labels": ["backend", "graph"],
        "description": """## 📋 Açıklama
Retrieve node'u uygulaması. VectorRetriever veya HybridRetriever'ı çağırır.

## 📝 Yapılacaklar
- [ ] retrieve_node() fonksiyonu oluşturulmuş
- [ ] VectorRetriever kullanılıyor (default)
- [ ] HybridRetriever seçeneği (config)
- [ ] retrieved_chunks state'e ekleniyor
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] Retrieve node çalışıyor
- [x] State güncelleniyor
- [x] Testler pass ediyor

## 🌿 Branch
feature/retrieve-node

## 📦 Bağımlılıklar
T12, T11

## 🔗 Engel Kaldırır
T14""",
    },
    {
        "number": "T14",
        "title": "Grade Node",
        "assignee": "burcuyldrm",
        "labels": ["backend", "graph"],
        "description": """## 📋 Açıklama
Retrieved chunks + query'yi LLM'e gönder. İlgililik sınıflandırması (Yes/No).

## 📝 Yapılacaklar
- [ ] grade_node() fonksiyonu oluşturulmuş
- [ ] LLM call'ı çalışıyor (OpenAI / Ollama)
- [ ] GradeResult döndürülüyor (relevant: bool, confidence: float, reasoning: str)
- [ ] State güncelleniyor
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] Grade node çalışıyor
- [x] LLM integration çalışıyor
- [x] Testler pass ediyor

## 🌿 Branch
feature/grade-node

## 📦 Bağımlılıklar
T13

## 🔗 Engel Kaldırır
T15""",
    },
    {
        "number": "T15",
        "title": "Conditional Edge Router",
        "assignee": "mervecalogluuu",
        "labels": ["backend", "graph"],
        "description": """## 📋 Açıklama
Grade sonucuna göre koşullu yönlendirme. confidence < THRESHOLD ve
iteration < 3 ise rewrite'a git, yoksa generate'a git.

## 📝 Yapılacaklar
- [ ] route_node() fonksiyonu oluşturulmuş
- [ ] GRADE_THRESHOLD config'de tanımlandı
- [ ] Koşul: confidence < threshold AND iteration < 3 → rewrite
- [ ] Aksi halde → generate
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] Router çalışıyor
- [x] Koşul mantığı doğru
- [x] Testler pass ediyor

## 🌿 Branch
feature/edge-router

## 📦 Bağımlılıklar
T14

## 🔗 Engel Kaldırır
T16, T17""",
    },
    {
        "number": "T16",
        "title": "Rewrite Node",
        "assignee": "mervecalogluuu",
        "labels": ["backend", "graph"],
        "description": """## 📋 Açıklama
LLM ile orijinal query'yi daha iyi bir arama terimine dönüştür. iteration sayacı artır.

## 📝 Yapılacaklar
- [ ] rewrite_node() fonksiyonu oluşturulmuş
- [ ] LLM query rewriting çalışıyor
- [ ] rewritten_query state'e ekleniyor
- [ ] iteration counter artıyor
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] Rewrite node çalışıyor
- [x] Query iyileştirilme çalışıyor
- [x] Testler pass ediyor

## 🌿 Branch
feature/rewrite-node

## 📦 Bağımlılıklar
T15

## 🔗 Engel Kaldırır
T13 (loop back)""",
    },
    {
        "number": "T17",
        "title": "Generate Node",
        "assignee": "burcuyldrm",
        "labels": ["backend", "graph"],
        "description": """## 📋 Açıklama
Geçerli chunks kullanarak son cevap üret. answer ve sources'ları state'e ekle.

## 📝 Yapılacaklar
- [ ] generate_node() fonksiyonu oluşturulmuş
- [ ] LLM answer generation çalışıyor
- [ ] final_answer state'e ekleniyor
- [ ] Sources listesi state'e ekleniyor
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] Generate node çalışıyor
- [x] Answer üretiliyor
- [x] Testler pass ediyor

## 🌿 Branch
feature/generate-node

## 📦 Bağımlılıklar
T15

## 🔗 Engel Kaldırır
T18""",
    },
    {
        "number": "T18",
        "title": "Kaynak Alıntı Formatı",
        "assignee": "burcuyldrm",
        "labels": ["backend", "graph"],
        "description": """## 📋 Açıklama
Yayın başlığı, yazarlar, ArXiv URL ve sayfa numarası ile alıntıları biçimlendir.

## 📝 Yapılacaklar
- [ ] citation.py dosyası oluşturulmuş
- [ ] format_citation() fonksiyonu oluşturulmuş
- [ ] Format: "Title (Authors, Page X) - URL"
- [ ] Markdown link formatı kullanılıyor
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] Citation formatter çalışıyor
- [x] Format doğru
- [x] Testler pass ediyor

## 🌿 Branch
feature/citation-formatter

## 📦 Bağımlılıklar
T17

## 🔗 Engel Kaldırır
T25""",
    },
    {
        "number": "T19",
        "title": "Standard RAG Baseline",
        "assignee": "mervecalogluuu",
        "labels": ["backend", "evaluation"],
        "description": """## 📋 Açıklama
StandardRAGChain: retrieve bir kez → generate. LangGraph'ı karşılaştırma için basit version.

## 📝 Yapılacaklar
- [ ] standard_rag.py dosyası oluşturulmuş
- [ ] StandardRAGChain sınıfı oluşturulmuş
- [ ] invoke() fonksiyonu çalışıyor
- [ ] Retrieve → Generate akışı
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] Standard RAG çalışıyor
- [x] Invoke fonksiyonu çalışıyor
- [x] Testler pass ediyor

## 🌿 Branch
feature/standard-rag

## 📦 Bağımlılıklar
T09, T11, T17

## 🔗 Engel Kaldırır
T20""",
    },
    {
        "number": "T20",
        "title": "RAGAS Evaluation Harness",
        "assignee": "burcuyldrm",
        "labels": ["backend", "evaluation"],
        "description": """## 📋 Açıklama
eval_runner.py: Standard RAG ve Auto-RAG'ı test Q&A dataset'i üzerinde çalıştır.
RAGAS ile skor: faithfulness, answer_relevancy, context_precision.

## 📝 Yapılacaklar
- [ ] eval_runner.py dosyası oluşturulmuş
- [ ] Test dataset yükleniyor
- [ ] StandardRAGChain ve LangGraph test ediliyor
- [ ] RAGAS metrikleri hesaplanıyor
- [ ] Sonuçlar JSON'a kaydediliyor
- [ ] Unit test yazılmış

## ✨ Kabul Kriterleri
- [x] RAGAS harness çalışıyor
- [x] Metrikler hesaplanıyor
- [x] Testler pass ediyor

## 🌿 Branch
feature/ragas-evaluation

## 📦 Bağımlılıklar
T19

## 🔗 Engel Kaldırır
T21, T22""",
    },
    {
        "number": "T21",
        "title": "Chunking Karşılaştırma Deneyi",
        "assignee": "mervecalogluuu",
        "labels": ["backend", "evaluation"],
        "description": """## 📋 Açıklama
CLI flag'ları: 512 vs 1024 karakterlik chunking. Sonuçları
results/chunking_experiment.json'a kaydet. 50 test sorgusu.

## 📝 Yapılacaklar
- [ ] Experiment runner CLI oluşturulmuş
- [ ] --chunk-size 512 ve 1024 seçenekleri
- [ ] 50 test sorgusu çalıştırılıyor
- [ ] RAGAS metrikleri hesaplanıyor (focus: context_precision)
- [ ] JSON sonuçları kaydediliyor

## ✨ Kabul Kriterleri
- [x] Chunking experiment çalışıyor
- [x] Sonuçlar kaydediliyor
- [x] Testler pass ediyor

## 🌿 Branch
feature/chunking-experiment

## 📦 Bağımlılıklar
T20

## 🔗 Engel Kaldırır
T26""",
    },
    {
        "number": "T22",
        "title": "Retrieval Karşılaştırma Deneyi",
        "assignee": "burcuyldrm",
        "labels": ["backend", "evaluation"],
        "description": """## 📋 Açıklama
CLI flag'ları: dense-only vs hybrid retrieval. Sonuçları
results/retrieval_experiment.json'a kaydet.

## 📝 Yapılacaklar
- [ ] --retrieval-mode dense ve hybrid seçenekleri
- [ ] 50 test sorgusu çalıştırılıyor
- [ ] RAGAS metrikleri hesaplanıyor (focus: answer_relevancy)
- [ ] JSON sonuçları kaydediliyor

## ✨ Kabul Kriterleri
- [x] Retrieval experiment çalışıyor
- [x] Sonuçlar kaydediliyor
- [x] Testler pass ediyor

## 🌿 Branch
feature/retrieval-experiment

## 📦 Bağımlılıklar
T20

## 🔗 Engel Kaldırır
T26""",
    },
    {
        "number": "T23",
        "title": "Streamlit App Skeleton",
        "assignee": "burcuyldrm",
        "labels": ["frontend"],
        "description": """## 📋 Açıklama
Streamlit app.py: 3-sütunlu layout:
- Sol: Query Input
- Orta: Thinking Process (trace)
- Sağ: Final Answer + Sources

## 📝 Yapılacaklar
- [ ] app.py dosyası oluşturulmuş
- [ ] 3-sütun layout st.columns ile
- [ ] Query input text box
- [ ] Thinking trace bölümü (placeholder)
- [ ] Answer bölümü (placeholder)
- [ ] Streamlit çalışıyor

## ✨ Kabul Kriterleri
- [x] Streamlit app çalışıyor
- [x] Layout hazır
- [x] UI responsive

## 🌿 Branch
feature/streamlit-skeleton

## 📦 Bağımlılıklar
T17

## 🔗 Engel Kaldırır
T24""",
    },
    {
        "number": "T24",
        "title": "Query Execution & Streaming Display",
        "assignee": "burcuyldrm",
        "labels": ["frontend"],
        "description": """## 📋 Açıklama
LangGraph runner'ı Streamlit'e bağla. Ara state'leri real-time göster.

## 📝 Yapılacaklar
- [ ] Query girdisinde rag_graph.invoke() çağrılıyor
- [ ] Intermediate state'ler gösteriliyor (retrieved chunks, grade, rewrite)
- [ ] Real-time update (st.write, st.container)
- [ ] Error handling

## ✨ Kabul Kriterleri
- [x] Query execution çalışıyor
- [x] Real-time update çalışıyor
- [x] Error handling var

## 🌿 Branch
feature/streamlit-execution

## 📦 Bağımlılıklar
T23, T12-T17

## 🔗 Engel Kaldırır
T25""",
    },
    {
        "number": "T25",
        "title": "Results Visualization Panel",
        "assignee": "burcuyldrm",
        "labels": ["frontend"],
        "description": """## 📋 Açıklama
Final answer, tıklanabilir alıntılar ve RAGAS bar chart göster.

## 📝 Yapılacaklar
- [ ] Final answer gösteriliyor
- [ ] Alıntılar clickable cards olarak
- [ ] Paper başlığı, sayfa, DOI link
- [ ] Similarity score gösteriliyor
- [ ] RAGAS metrikleri bar chart (plotly)

## ✨ Kabul Kriterleri
- [x] Results panel çalışıyor
- [x] Citations gösteriliyor
- [x] Metrics görülüyor

## 🌿 Branch
feature/results-visualization

## 📦 Bağımlılıklar
T24, T18

## 🔗 Engel Kaldırır
Hiç""",
    },
    {
        "number": "T26",
        "title": "Experiment Comparison Dashboard",
        "assignee": "burcuyldrm",
        "labels": ["frontend"],
        "description": """## 📋 Açıklama
İkinci Streamlit sayfası: results/*.json yükle, E1/E2/E3 karşılaştır.
Tablo ve grafikler (plotly).

## 📝 Yapılacaklar
- [ ] Multi-page app (st.navigation veya sidebar)
- [ ] results/ klasöründen JSON yükleniyor
- [ ] E1 (chunking) table + chart
- [ ] E2 (retrieval) table + chart
- [ ] E3 (iteration) table + chart
- [ ] Karşılaştırma grafiği

## ✨ Kabul Kriterleri
- [x] Dashboard çalışıyor
- [x] Grafikler gösteriliyor
- [x] Multi-page setup var

## 🌿 Branch
feature/experiments-dashboard

## 📦 Bağımlılıklar
T21, T22

## 🔗 Engel Kaldırır
Hiç""",
    },
    {
        "number": "T27",
        "title": "Unit Tests: Data Layer",
        "assignee": "mervecalogluuu",
        "labels": ["testing"],
        "description": """## 📋 Açıklama
Fetcher'lar, PDF extractor'ı ve chunker'ı test et.

## 📝 Yapılacaklar
- [ ] test_arxiv_fetcher.py oluşturulmuş (min 5 test)
- [ ] test_core_fetcher.py oluşturulmuş (min 5 test)
- [ ] test_pdf_extractor.py oluşturulmuş (min 5 test)
- [ ] test_chunker.py oluşturulmuş (min 5 test)
- [ ] pytest çalıştırıldığında %80+ pass

## ✨ Kabul Kriterleri
- [x] Tüm testler yazılmış
- [x] %80+ pass oranı
- [x] Mock'lar kullanılıyor

## 🌿 Branch
test/data-layer-tests

## 📦 Bağımlılıklar
T02-T06

## 🔗 Engel Kaldırır
Hiç""",
    },
    {
        "number": "T28",
        "title": "Integration Test: Full Pipeline",
        "assignee": "mervecalogluuu",
        "labels": ["testing"],
        "description": """## 📋 Açıklama
3 örnek sorgu üzerinde full LangGraph pipeline test et. Answer ve sources
varlığını assert et.

## 📝 Yapılacaklar
- [ ] test_full_pipeline.py oluşturulmuş
- [ ] 3 sorgu üzerinde end-to-end test
- [ ] Assert: final_answer != None
- [ ] Assert: sources != []
- [ ] Test geçiyor

## ✨ Kabul Kriterleri
- [x] Integration test çalışıyor
- [x] Assertions doğru
- [x] Test geçiyor

## 🌿 Branch
test/integration-tests

## 📦 Bağımlılıklar
T12-T17

## 🔗 Engel Kaldırır
Hiç""",
    },
]

print("🚀 GitHub Issues oluşturuluyor...\n")
created_count = 0

for task in TASKS:
    try:
        issue = repo.create_issue(
            title=f"{task['number']} - {task['title']}",
            body=task["description"],
            assignee=task["assignee"],
            labels=task["labels"],
        )
        print(f"✅ {task['number']} - {task['title']}")
        print(f"   🔗 {issue.html_url}\n")
        created_count += 1
    except Exception as e:
        print(f"❌ {task['number']} - HATA: {str(e)}\n")

print(f"\n{'='*60}")
print(f"✨ Toplam {created_count}/{len(TASKS)} Issue başarıyla oluşturuldu!")
print(f"{'='*60}")


