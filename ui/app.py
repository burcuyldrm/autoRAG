"""
Auto-RAG Streamlit App

Ana sayfa: Query input, thinking trace ve final answer (3 sütun layout).
Sidebar navigasyonu ile Experiment Dashboard'a geçiş.

Çalıştırma: streamlit run ui/app.py
"""
import json
import sys
import os

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="Auto-RAG",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Auto-RAG")
page = st.sidebar.radio("Sayfa", ["RAG Sorgusu", "Deney Karşılaştırma"])

# ---------------------------------------------------------------------------
# Page: RAG Sorgusu (T23 + T24 + T25)
# ---------------------------------------------------------------------------
if page == "RAG Sorgusu":
    st.title("Auto-RAG — Bilimsel Literatür Sorgulama")

    col_input, col_trace, col_answer = st.columns([1, 1, 1])

    # --- Sol: Query Input ---
    with col_input:
        st.subheader("Sorgu")
        query = st.text_area("Sorunuzu girin:", height=150, placeholder="Örn: What is retrieval augmented generation?")
        run_btn = st.button("Çalıştır", type="primary", use_container_width=True)
        retrieval_mode = st.selectbox("Retrieval modu", ["hybrid", "dense"])

    # --- Orta: Thinking Process ---
    with col_trace:
        st.subheader("Düşünce Süreci")
        trace_container = st.container()

    # --- Sağ: Final Answer + Sources ---
    with col_answer:
        st.subheader("Sonuç")
        answer_container = st.container()

    # --- Query Execution (T24) ---
    if run_btn and query.strip():
        with st.spinner("Çalışıyor..."):
            try:
                result = _run_rag_query(query, retrieval_mode)
            except Exception as exc:
                st.error(f"Hata oluştu: {exc}")
                result = None

        if result:
            # Thinking trace
            with trace_container:
                if result.get("retrieved_chunks"):
                    with st.expander("Retrieve — Getirilen Parçalar", expanded=True):
                        for i, chunk in enumerate(result["retrieved_chunks"][:3], 1):
                            st.markdown(f"**[{i}]** {chunk.get('text', '')[:200]}...")
                if result.get("grade_result"):
                    grade = result["grade_result"]
                    icon = "✅" if grade.get("relevant") else "❌"
                    with st.expander(f"Grade — Alaka Değerlendirmesi {icon}", expanded=True):
                        st.write(f"**İlgili:** {grade.get('relevant')}")
                        st.write(f"**Güven:** {grade.get('confidence', 0):.0%}")
                        st.write(f"**Gerekçe:** {grade.get('reasoning', '')}")
                if result.get("rewritten_query"):
                    with st.expander("Rewrite — Yeniden Yazılan Sorgu"):
                        st.info(result["rewritten_query"])

            # Final answer + sources (T25)
            with answer_container:
                st.markdown("#### Cevap")
                st.write(result.get("final_answer", "Cevap üretilemedi."))

                sources = result.get("sources", [])
                if sources:
                    st.markdown("#### Kaynaklar")
                    from graph.citation import format_sources
                    citations = format_sources(sources)
                    for citation in citations:
                        st.markdown(f"- {citation}")

                    if result.get("ragas_metrics"):
                        st.markdown("#### RAGAS Metrikleri")
                        metrics = result["ragas_metrics"]
                        cols = st.columns(len(metrics))
                        for col, (name, val) in zip(cols, metrics.items()):
                            col.metric(name.replace("_", " ").title(), f"{val:.2f}")

    elif run_btn:
        st.warning("Lütfen bir sorgu girin.")

# ---------------------------------------------------------------------------
# Page: Deney Karşılaştırma (T26)
# ---------------------------------------------------------------------------
else:
    _render_dashboard()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _run_rag_query(query: str, retrieval_mode: str) -> dict:
    """Runs the LangGraph pipeline and returns state."""
    try:
        from graph.state import GraphState
        from graph.nodes.grade_node import grade_node
        from graph.nodes.generate_node import generate_node

        # Pipeline bağımlılıkları (T09, T11, T13, T15, T16) henüz tamamlanmadığında
        # minimal stub döndür — T09/T11/T13-T16 merge edilince gerçek pipeline bağlanır
        state: GraphState = {
            "query": query,
            "chunks": _stub_retrieve(query),
            "iteration": 0,
        }
        state = grade_node(state, llm=_get_llm())
        state = generate_node(state, llm=_get_llm())
        return state
    except Exception as exc:
        return {"final_answer": f"Pipeline hatası: {exc}", "sources": []}


def _stub_retrieve(query: str) -> list[dict]:
    """Placeholder until T09/T11 retriever nodes are merged."""
    return [
        {
            "id": "stub-1",
            "text": f"[Stub] '{query}' için getirilen örnek metin parçası.",
            "metadata": {"paper_id": "stub", "page": 1, "title": "Örnek Makale"},
        }
    ]


def _get_llm():
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


def _render_dashboard() -> None:
    """T26 — Experiment Comparison Dashboard."""
    import plotly.graph_objects as go

    st.title("Deney Karşılaştırma Paneli")

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")] if os.path.isdir(results_dir) else []

    if not json_files:
        st.info("Henüz deney sonucu yok. `python -m eval.eval_runner` veya `python -m eval.retrieval_experiment` çalıştırın.")
        return

    selected = st.multiselect("Sonuç dosyaları:", json_files, default=json_files[:3])
    if not selected:
        return

    experiments = {}
    for fname in selected:
        try:
            with open(os.path.join(results_dir, fname)) as f:
                experiments[fname] = json.load(f)
        except Exception:
            st.warning(f"{fname} okunamadı.")

    # --- Metrics table ---
    rows = []
    for fname, data in experiments.items():
        row = {"Deney": fname}
        for key in ("standard_rag", "auto_rag"):
            if key in data:
                for metric, val in data[key].items():
                    row[f"{key}/{metric}"] = round(val, 3)
        for key in ("faithfulness", "answer_relevancy", "context_precision"):
            if key in data:
                row[key] = round(data[key], 3)
        rows.append(row)

    if rows:
        st.subheader("Metrik Tablosu")
        st.dataframe(rows, use_container_width=True)

    # --- Bar chart ---
    metric_keys = [k for k in rows[0] if k != "Deney"] if rows else []
    if metric_keys:
        selected_metric = st.selectbox("Grafik için metrik:", metric_keys)
        fig = go.Figure(
            data=[
                go.Bar(
                    name=r["Deney"],
                    x=[r["Deney"]],
                    y=[r.get(selected_metric, 0)],
                )
                for r in rows
            ]
        )
        fig.update_layout(
            title=f"{selected_metric} Karşılaştırması",
            yaxis=dict(range=[0, 1]),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Retrieval experiment comparison ---
    retrieval_exps = {k: v for k, v in experiments.items() if "retrieval_mode" in v}
    if retrieval_exps:
        st.subheader("Retrieval Karşılaştırması (T22)")
        labels = list(retrieval_exps.keys())
        scores = [v.get("answer_relevancy", 0) for v in retrieval_exps.values()]
        modes = [v.get("retrieval_mode", "?") for v in retrieval_exps.values()]
        fig2 = go.Figure(
            data=[go.Bar(x=[f"{m}: {l}" for m, l in zip(modes, labels)], y=scores)]
        )
        fig2.update_layout(title="Answer Relevancy — Dense vs Hybrid", yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig2, use_container_width=True)
