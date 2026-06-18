"""
Auto-RAG Streamlit App

Sayfalar:
  1. RAG Sorgusu       — sorgu, düşünce izi, cevap
  2. Metrics Dashboard — results/*.json dosyalarından RAGAS metrikleri
  3. Model Comparison  — model bazlı karşılaştırma
  4. Benchmark Results — rag / retriever / chunk / topk benchmark'ları
  5. Retrieved Sources — son sorgudaki chunk'lar ve atıflar

Çalıştırma: streamlit run ui/app.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---------------------------------------------------------------------------
# Sayfa konfigürasyonu
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Auto-RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

_RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

_METRIC_LABELS: dict[str, str] = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "avg_latency_seconds": "Avg Latency (s)",
    "avg_rewrite_count": "Avg Rewrites",
}

_COLORS = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#B279A2", "#FF9DA7"]

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "last_query_result" not in st.session_state:
    st.session_state["last_query_result"] = None
if "last_query_text" not in st.session_state:
    st.session_state["last_query_text"] = ""

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Auto-RAG")
st.sidebar.caption("Self-Reflective Academic RAG")

PAGES = [
    "RAG Sorgusu",
    "Metrics Dashboard",
    "Model Comparison",
    "Benchmark Results",
    "Retrieved Sources",
]
page = st.sidebar.radio("Sayfa", PAGES)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_all_results() -> dict[str, dict]:
    """results/ klasöründeki tüm JSON dosyalarını yükler."""
    if not os.path.isdir(_RESULTS_DIR):
        return {}
    data: dict[str, dict] = {}
    for fname in sorted(os.listdir(_RESULTS_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(_RESULTS_DIR, fname)) as f:
                data[fname] = json.load(f)
        except Exception:
            pass
    return data


def _extract_experiments(all_data: dict[str, dict]) -> list[dict[str, Any]]:
    """Tüm JSON dosyalarından ExperimentResult satırlarını çıkartır."""
    rows: list[dict] = []
    for fname, data in all_data.items():
        exps = data.get("experiments", [])
        if exps:
            for exp in exps:
                row = {k: v for k, v in exp.items()}
                row.setdefault("_source_file", fname)
                rows.append(row)
        else:
            # Eski düz format (örn. retrieval_experiment çıktısı)
            if any(k in data for k in ("faithfulness", "answer_relevancy", "experiment_type")):
                row = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
                row.setdefault("experiment_name", fname.replace(".json", ""))
                row.setdefault("_source_file", fname)
                rows.append(row)
    return rows


def _safe_float(val: Any, decimals: int = 3) -> str:
    """Sayısal değeri güvenli biçimde formatlar."""
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def _no_results_hint(experiment_type: str | None = None) -> None:
    """Sonuç dosyası yokken hangi komutu çalıştıracağını gösterir."""
    cmds: dict[str | None, str] = {
        "comparison": "python -m eval.benchmark_runner --dataset data/qa.json --benchmark rag",
        "retrieval":  "python -m eval.benchmark_runner --dataset data/qa.json --benchmark retriever",
        "chunking":   "python -m eval.benchmark_runner --dataset data/qa.json --benchmark chunk",
        "topk":       "python -m eval.topk_experiment   --dataset data/qa.json",
        "model_comparison": "python -m eval.model_comparison --dataset data/qa.json",
        None: (
            "python -m eval.benchmark_runner --dataset data/qa.json\n"
            "python -m eval.model_comparison --dataset data/qa.json\n"
            "python -m eval.topk_experiment  --dataset data/qa.json"
        ),
    }
    cmd = cmds.get(experiment_type, cmds[None])
    st.info(
        "Henüz deney sonucu bulunamadı.\n\n"
        f"Deneyi çalıştırmak için:\n```bash\n{cmd}\n```"
    )


# ---------------------------------------------------------------------------
# Chart helper
# ---------------------------------------------------------------------------

def _bar_chart(
    rows: list[dict],
    x_field: str,
    y_fields: list[str],
    title: str,
    y_range: list[float] | None = None,
    height: int = 380,
) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, y_field in enumerate(y_fields):
        xs, ys = [], []
        for row in rows:
            val = row.get(y_field)
            if val is not None:
                xs.append(str(row.get(x_field, "?")))
                ys.append(float(val))
        if xs:
            fig.add_trace(go.Bar(
                name=_METRIC_LABELS.get(y_field, y_field),
                x=xs,
                y=ys,
                marker_color=_COLORS[i % len(_COLORS)],
            ))
    fig.update_layout(
        title=title,
        barmode="group",
        yaxis=dict(range=y_range or [0, 1.05]),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 1 — RAG Sorgusu
# ---------------------------------------------------------------------------

def _page_rag_query() -> None:
    st.title("Auto-RAG — Bilimsel Literatür Sorgulama")

    col_input, col_trace, col_answer = st.columns([1, 1, 1])

    with col_input:
        st.subheader("Sorgu")
        query = st.text_area(
            "Sorunuzu girin:",
            height=150,
            placeholder="Örn: What is retrieval augmented generation?",
        )
        run_btn = st.button("Çalıştır", type="primary", use_container_width=True)
        retrieval_mode = st.selectbox("Retrieval modu", ["hybrid", "dense", "bm25"])
        top_k = st.slider("Top-k", min_value=1, max_value=20, value=5)

    with col_trace:
        st.subheader("Düşünce Süreci")
        trace_container = st.container()

    with col_answer:
        st.subheader("Sonuç")
        answer_container = st.container()

    if run_btn and query.strip():
        with st.spinner("Çalışıyor..."):
            try:
                result = _run_rag_query(query, retrieval_mode, top_k)
                st.session_state["last_query_result"] = result
                st.session_state["last_query_text"] = query
            except Exception as exc:
                st.error(f"Hata oluştu: {exc}")
                result = None

        if result:
            with trace_container:
                chunks = result.get("chunks") or result.get("retrieved_chunks", [])
                if chunks:
                    with st.expander("Retrieve — Getirilen Parçalar", expanded=True):
                        scores = result.get("retrieved_scores", [])
                        for i, chunk in enumerate(chunks[:5], 1):
                            score_str = (
                                f" (skor: {scores[i-1]:.3f})"
                                if i - 1 < len(scores) else ""
                            )
                            meta = chunk.get("metadata", {})
                            title_str = meta.get("title", "")
                            title_line = f"*{title_str}*  " if title_str else ""
                            st.markdown(
                                f"**[{i}]**{score_str} {title_line}"
                                f"{chunk.get('text', '')[:200]}..."
                            )

                grade_result = result.get("grade_result")
                grade_relevant = result.get(
                    "grade_relevant",
                    grade_result.get("relevant") if grade_result else None,
                )
                grade_confidence = result.get(
                    "grade_confidence",
                    grade_result.get("confidence", 0.0) if grade_result else 0.0,
                )
                grade_reasoning = grade_result.get("reasoning", "") if grade_result else ""

                if grade_relevant is not None:
                    icon = "✅" if grade_relevant else "❌"
                    with st.expander(f"Grade — Alaka Değerlendirmesi {icon}", expanded=True):
                        st.write(f"**İlgili:** {grade_relevant}")
                        st.write(f"**Güven:** {grade_confidence:.0%}")
                        if grade_reasoning:
                            st.write(f"**Gerekçe:** {grade_reasoning}")

                if result.get("rewritten_query"):
                    rewrite_count = result.get("rewrite_count", 1)
                    with st.expander(f"Rewrite — Yeniden Yazılan Sorgu ({rewrite_count}x)"):
                        st.info(result["rewritten_query"])

            with answer_container:
                answer = result.get("answer") or result.get("final_answer", "Cevap üretilemedi.")
                st.markdown("#### Cevap")
                st.write(answer)

                latency = result.get("latency_seconds")
                if latency is not None:
                    st.caption(
                        f"Süre: {latency:.2f}s  |  "
                        f"Retrieval: {result.get('retrieval_mode', '—')}  |  "
                        f"top_k: {result.get('top_k', '—')}"
                    )

                sources = result.get("sources", [])
                if sources:
                    st.markdown("#### Kaynaklar")
                    try:
                        from graph.citation import format_sources
                        for citation in format_sources(sources):
                            st.markdown(f"- {citation}")
                    except Exception:
                        for s in sources:
                            st.markdown(f"- {s.get('metadata', {}).get('title', s.get('id', '?'))}")

                st.caption("Retrieved Sources sayfasında chunk detaylarını görüntüleyebilirsiniz.")

    elif run_btn:
        st.warning("Lütfen bir sorgu girin.")


# ---------------------------------------------------------------------------
# Page 2 — Metrics Dashboard
# ---------------------------------------------------------------------------

def _page_metrics_dashboard() -> None:
    st.title("Metrics Dashboard")

    all_data = _load_all_results()
    if not all_data:
        _no_results_hint()
        return

    json_files = list(all_data.keys())
    selected = st.multiselect(
        "Sonuç dosyaları:", json_files,
        default=json_files[:min(5, len(json_files))],
    )
    if not selected:
        st.info("En az bir dosya seçin.")
        return

    rows = _extract_experiments({k: all_data[k] for k in selected if k in all_data})
    if not rows:
        st.warning("Seçili dosyalarda experiment verisi bulunamadı.")
        return

    # --- Metrik tablosu ---
    st.subheader("Metrik Tablosu")
    table_cols = [
        "experiment_name", "experiment_type",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "avg_latency_seconds", "avg_rewrite_count", "n_questions", "_source_file",
    ]
    display_rows = []
    for row in rows:
        display_rows.append({
            col: _safe_float(row.get(col)) if col in _METRIC_LABELS else row.get(col, "—")
            for col in table_cols
        })
    st.dataframe(display_rows, use_container_width=True)

    # --- Grafikler ---
    st.subheader("Metrik Grafikleri")

    ragas_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ragas_rows = [r for r in rows if any(r.get(m) is not None for m in ragas_metrics)]
    if ragas_rows:
        _bar_chart(
            ragas_rows,
            x_field="experiment_name",
            y_fields=ragas_metrics,
            title="RAGAS Metrikleri",
            y_range=[0, 1.05],
        )

    latency_rows = [r for r in rows if r.get("avg_latency_seconds") is not None]
    if latency_rows:
        _bar_chart(
            latency_rows,
            x_field="experiment_name",
            y_fields=["avg_latency_seconds"],
            title="Ortalama Gecikme (saniye)",
            y_range=None,
            height=300,
        )

    rewrite_rows = [r for r in rows if r.get("avg_rewrite_count") not in (None, 0)]
    if rewrite_rows:
        _bar_chart(
            rewrite_rows,
            x_field="experiment_name",
            y_fields=["avg_rewrite_count"],
            title="Ortalama Rewrite Sayısı",
            y_range=None,
            height=300,
        )


# ---------------------------------------------------------------------------
# Page 3 — Model Comparison
# ---------------------------------------------------------------------------

def _page_model_comparison() -> None:
    st.title("Model Comparison")

    all_data = _load_all_results()
    rows = _extract_experiments(all_data)
    model_rows = [r for r in rows if r.get("experiment_type") == "model_comparison"]

    if not model_rows:
        _no_results_hint("model_comparison")
        return

    # --- Filtre ---
    all_models = sorted({r.get("model_name", "?") for r in model_rows if r.get("model_name")})
    selected_models = st.multiselect("Model filtresi:", all_models, default=all_models)
    filtered = [r for r in model_rows if r.get("model_name") in selected_models]
    if not filtered:
        st.info("Filtre sonucu boş.")
        return

    # --- Tablo ---
    st.subheader("Model Karşılaştırma Tablosu")
    table_cols = [
        "model_name", "provider",
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "avg_latency_seconds", "token_usage", "cost_estimate",
        "n_questions",
    ]
    display_rows = []
    for row in filtered:
        display_rows.append({
            col: _safe_float(row.get(col)) if col in _METRIC_LABELS or col in (
                "token_usage", "cost_estimate"
            ) else row.get(col, "—")
            for col in table_cols
        })
    st.dataframe(display_rows, use_container_width=True)

    # --- Bar chart ---
    st.subheader("Metrik Grafikleri")
    metric_opts = [m for m in ["faithfulness", "answer_relevancy", "avg_latency_seconds"] if
                   any(r.get(m) is not None for r in filtered)]
    if metric_opts:
        _bar_chart(
            filtered,
            x_field="model_name",
            y_fields=metric_opts,
            title="Model Bazlı Metrikler",
        )


# ---------------------------------------------------------------------------
# Page 4 — Benchmark Results
# ---------------------------------------------------------------------------

def _page_benchmark_results() -> None:
    st.title("Benchmark Results")

    all_data = _load_all_results()
    rows = _extract_experiments(all_data)

    tabs = st.tabs([
        "Standard RAG vs Auto-RAG",
        "Retriever Karşılaştırması",
        "Chunk Size Karşılaştırması",
        "Top-k Karşılaştırması",
    ])

    # --- Tab 1: Standard vs Auto-RAG ---
    with tabs[0]:
        comp_rows = [r for r in rows if r.get("experiment_type") == "comparison"]
        if not comp_rows:
            _no_results_hint("comparison")
        else:
            st.subheader("Standard RAG vs Auto-RAG")
            _benchmark_table_and_chart(
                comp_rows,
                x_field="experiment_name",
                title="Standard RAG vs Auto-RAG — RAGAS Metrikleri",
            )
            rw_rows = [r for r in comp_rows if r.get("avg_rewrite_count") is not None]
            if rw_rows:
                _bar_chart(
                    rw_rows,
                    x_field="experiment_name",
                    y_fields=["avg_rewrite_count", "avg_latency_seconds"],
                    title="Rewrite Sayısı & Gecikme",
                    y_range=None,
                    height=300,
                )

    # --- Tab 2: Retriever ---
    with tabs[1]:
        ret_rows = [r for r in rows if r.get("experiment_type") == "retrieval"]
        if not ret_rows:
            _no_results_hint("retrieval")
        else:
            st.subheader("BM25 vs Dense vs Hybrid Retrieval")
            _benchmark_table_and_chart(
                ret_rows,
                x_field="retriever_type",
                title="Retriever Tipi — RAGAS Metrikleri",
            )

    # --- Tab 3: Chunk Size ---
    with tabs[2]:
        chunk_rows = [r for r in rows if r.get("experiment_type") == "chunking"]
        if not chunk_rows:
            _no_results_hint("chunking")
        else:
            chunk_rows = sorted(chunk_rows, key=lambda r: r.get("chunk_size") or 0)
            st.subheader("Chunk Size Karşılaştırması (256 / 512 / 1024 / 2048)")
            _benchmark_table_and_chart(
                chunk_rows,
                x_field="chunk_size",
                title="Chunk Size — RAGAS Metrikleri",
            )

    # --- Tab 4: Top-k ---
    with tabs[3]:
        topk_rows = [r for r in rows if r.get("experiment_type") == "topk"]
        if not topk_rows:
            _no_results_hint("topk")
        else:
            topk_rows = sorted(topk_rows, key=lambda r: r.get("top_k") or 0)
            st.subheader("Top-k Karşılaştırması (3 / 5 / 10 / 15)")
            _benchmark_table_and_chart(
                topk_rows,
                x_field="top_k",
                title="Top-k — RAGAS Metrikleri",
            )


def _benchmark_table_and_chart(rows: list[dict], x_field: str, title: str) -> None:
    table_cols = [
        x_field, "faithfulness", "answer_relevancy",
        "context_precision", "context_recall",
        "avg_latency_seconds", "avg_rewrite_count", "n_questions",
    ]
    display = []
    for row in rows:
        display.append({
            col: _safe_float(row.get(col)) if col in _METRIC_LABELS else row.get(col, "—")
            for col in table_cols
        })
    st.dataframe(display, use_container_width=True)

    ragas_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    if any(r.get(m) is not None for r in rows for m in ragas_metrics):
        _bar_chart(rows, x_field=x_field, y_fields=ragas_metrics, title=title)


# ---------------------------------------------------------------------------
# Page 5 — Retrieved Sources
# ---------------------------------------------------------------------------

def _page_retrieved_sources() -> None:
    st.title("Retrieved Sources")

    result = st.session_state.get("last_query_result")
    query_text = st.session_state.get("last_query_text", "")

    if result is None:
        st.info(
            "Henüz bir sorgu çalıştırılmadı. "
            "**RAG Sorgusu** sayfasından bir sorgu gönderin."
        )
        return

    st.markdown(f"**Son sorgu:** `{query_text}`")
    st.caption(
        f"Retrieval: {result.get('retrieval_mode', '—')}  |  "
        f"top_k: {result.get('top_k', '—')}  |  "
        f"Süre: {_safe_float(result.get('latency_seconds'), 2)}s  |  "
        f"Rewrite sayısı: {result.get('rewrite_count', 0)}"
    )
    st.divider()

    chunks = result.get("chunks") or result.get("retrieved_chunks", [])
    scores = result.get("retrieved_scores", [])

    if not chunks:
        st.warning("Bu sorgu için chunk bulunamadı.")
        return

    st.subheader(f"Getirilen Chunk'lar ({len(chunks)} adet)")
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        score = scores[i] if i < len(scores) else None
        score_badge = f"Skor: `{score:.4f}`" if score is not None else ""
        title = meta.get("title", meta.get("paper_id", chunk.get("id", f"Chunk {i+1}")))

        with st.expander(f"[{i+1}] {title}  {score_badge}", expanded=i == 0):
            st.markdown("**Metin:**")
            st.text(chunk.get("text", ""))

            cols = st.columns(3)
            cols[0].markdown(f"**ID:** `{chunk.get('id', '—')}`")
            cols[1].markdown(f"**Sayfa:** {meta.get('page', '—')}")
            cols[2].markdown(f"**Kaynak:** {meta.get('source', meta.get('paper_id', '—'))}")

            url = meta.get("url") or meta.get("pdf_url") or meta.get("html_url")
            if url:
                st.markdown(f"[Makaleyi Aç]({url})")

    sources = result.get("sources", [])
    if sources:
        st.divider()
        st.subheader("Atıf Listesi")
        try:
            from graph.citation import format_sources
            for citation in format_sources(sources):
                st.markdown(f"- {citation}")
        except Exception:
            for s in sources:
                st.markdown(f"- `{s.get('id', '?')}` — {s.get('metadata', {}).get('title', '')}")


# ---------------------------------------------------------------------------
# Pipeline helpers (RAG Sorgusu sayfası için)
# ---------------------------------------------------------------------------

def _run_rag_query(query: str, retrieval_mode: str, top_k: int = 5) -> dict:
    """Auto-RAG pipeline'ını çalıştırır ve sonuç dict'ini döndürür."""
    try:
        from graph.autorag_chain import AutoRAGChain

        chain = AutoRAGChain(
            vectorstore=None,
            bm25_retriever=None,
            llm=_get_llm(),
            grade_threshold=0.60,
            max_rewrites=2,
        )
        result = chain.run(query, retrieval_mode=retrieval_mode, top_k=top_k)
        if not result.get("chunks"):
            result["chunks"] = _stub_retrieve(query)
        return result
    except Exception as exc:
        return {"answer": f"Pipeline hatası: {exc}", "sources": [], "chunks": []}


def _stub_retrieve(query: str) -> list[dict]:
    """Gerçek retriever bağlanana kadar kullanılan örnek chunk."""
    return [
        {
            "id": "stub-1",
            "text": f"[Stub] '{query}' için getirilen örnek metin parçası.",
            "metadata": {"paper_id": "stub", "page": 1, "title": "Örnek Makale"},
        }
    ]


def _get_llm():
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Sayfa yönlendirme
# ---------------------------------------------------------------------------

if page == "RAG Sorgusu":
    _page_rag_query()
elif page == "Metrics Dashboard":
    _page_metrics_dashboard()
elif page == "Model Comparison":
    _page_model_comparison()
elif page == "Benchmark Results":
    _page_benchmark_results()
elif page == "Retrieved Sources":
    _page_retrieved_sources()
