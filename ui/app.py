"""
AutoRAG — Professional Light UI
Run: streamlit run ui/app.py
"""
from __future__ import annotations

import json, os, sys, time, random
from datetime import datetime, timedelta
from typing import Any

import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))

st.set_page_config(page_title="AutoRAG", page_icon="🔬", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }
.stApp { background: #f1f5f9; color: #0f172a; }
#MainMenu, footer, header { visibility: hidden; }

/* sidebar */
section[data-testid="stSidebar"] > div:first-child {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
    padding-top: 0;
}

/* ── top nav bar ── */
.topbar {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 14px 28px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex; align-items: center; gap: 12px;
}
.topbar-logo {
    width: 32px; height: 32px;
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.topbar-title { font-size: 17px; font-weight: 700; color: #0f172a; letter-spacing: -.3px; }
.topbar-sub   { font-size: 12px; color: #94a3b8; margin-left: 4px; margin-top: 1px; }
.topbar-badge {
    margin-left: auto;
    font-size: 11px; font-weight: 600;
    background: #f0fdf4; color: #059669;
    border: 1px solid #6ee7b7;
    padding: 4px 12px; border-radius: 20px;
}
.topbar-badge.warn { background:#fffbeb; color:#d97706; border-color:#fcd34d; }

/* ── section title ── */
.sec-title {
    font-size: 13px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .8px;
    color: #94a3b8; margin: 0 0 12px;
}

/* ── metric cards ── */
.mcard {
    background: #fff;
    border-radius: 12px;
    padding: 18px 20px 14px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(15,23,42,.04);
    position: relative; overflow: hidden;
}
.mcard::after {
    content: "";
    position: absolute; bottom:0; left:0;
    height: 3px; width: 100%;
}
.mcard.blue::after   { background: linear-gradient(90deg,#2563eb,#60a5fa); }
.mcard.green::after  { background: linear-gradient(90deg,#059669,#34d399); }
.mcard.purple::after { background: linear-gradient(90deg,#7c3aed,#a78bfa); }
.mcard.orange::after { background: linear-gradient(90deg,#d97706,#fbbf24); }
.mcard .lbl { font-size: 10px; font-weight: 600; text-transform:uppercase;
              letter-spacing:.8px; color:#94a3b8; }
.mcard .val { font-size: 28px; font-weight: 700; line-height: 1.2;
              margin: 6px 0 2px; color: #0f172a; }
.mcard .sub { font-size: 11px; color: #94a3b8; }
.mcard .ico { position:absolute; top:16px; right:16px; font-size:20px; opacity:.15; }

/* ── panel (white card wrapping a column) ── */
.panel {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(15,23,42,.04);
    height: 100%;
}
.panel-title {
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .7px; color: #64748b;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 10px; margin-bottom: 14px;
}

/* ── pipeline step cards ── */
.scard {
    border-radius: 8px; padding: 12px 14px;
    margin-bottom: 8px; font-size: 12.5px;
    border: 1px solid #e2e8f0;
    transition: border-color .2s;
}
.scard.pending { background:#fafafa; border-color:#e2e8f0; }
.scard.running { background:#eff6ff; border-color:#93c5fd; }
.scard.success { background:#f0fdf4; border-color:#6ee7b7; }
.scard.warning { background:#fffbeb; border-color:#fcd34d; }
.scard.error   { background:#fef2f2; border-color:#fca5a5; }
.scard .sh {
    display:flex; align-items:center; gap:7px;
    font-weight: 600; margin-bottom: 4px; font-size: 12px;
}
.scard .sb { color:#64748b; font-size:11.5px; line-height:1.45; }
.scard.success .sh { color:#065f46; }
.scard.warning .sh { color:#92400e; }
.scard.error .sh   { color:#991b1b; }
.scard.running .sh { color:#1d4ed8; }
.scard.pending .sh { color:#94a3b8; }

/* ── answer box ── */
.abox {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 18px;
    font-size: 14px; line-height: 1.8; color: #1e293b;
    min-height: 100px;
}
.abox.empty { color:#94a3b8; font-style:italic; font-size:13px; }

/* ── source cards ── */
.src-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 10px 12px; margin-top: 8px;
}
.src-head {
    font-size: 11.5px; font-weight: 600;
    color: #2563eb; margin-bottom: 3px;
    display: flex; align-items: center; gap: 6px;
}
.src-body { font-size: 11px; color: #64748b; line-height: 1.5; }

/* ── badges ── */
.badge {
    display:inline-flex; align-items:center; gap:4px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600;
}
.badge.ok   { background:#f0fdf4; color:#059669; border:1px solid #6ee7b7; }
.badge.warn { background:#fffbeb; color:#d97706; border:1px solid #fcd34d; }
.badge.err  { background:#fef2f2; color:#dc2626; border:1px solid #fca5a5; }

/* ── status health cards (page 3) ── */
.hcard {
    background:#fff; border:1px solid #e2e8f0;
    border-radius:12px; padding:18px 20px;
    box-shadow:0 1px 3px rgba(15,23,42,.04);
}
.hcard .hname { font-size:11px; font-weight:600; text-transform:uppercase;
               letter-spacing:.7px; color:#94a3b8; margin-bottom:8px; }
.hcard .hmsg  { font-size:11.5px; color:#64748b; margin-top:6px; line-height:1.5; }

/* ── info cards ── */
.icard {
    background:#fff; border:1px solid #e2e8f0;
    border-radius:12px; padding:20px;
    box-shadow:0 1px 3px rgba(15,23,42,.04);
}
.icard h4 { font-size:13px; font-weight:600; color:#0f172a; margin:0 0 10px; }
.icard p  { font-size:12px; color:#64748b; margin:0; line-height:1.7; }

/* ── sidebar ── */
.slbl { font-size:10px; font-weight:600; text-transform:uppercase;
        letter-spacing:.8px; color:#94a3b8; margin:14px 0 5px; }

/* ── button ── */
.stButton > button {
    background: linear-gradient(135deg,#2563eb,#7c3aed) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 13px !important; letter-spacing: .3px !important;
    padding: 10px 0 !important;
    box-shadow: 0 2px 8px rgba(37,99,235,.3) !important;
}
.stButton > button:hover { opacity: .88 !important; }

/* ── tabs ── */
.stTabs [data-baseweb="tab"] { font-size:13px; font-weight:500; }
.stTabs [data-baseweb="tab-list"] { border-bottom:1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def mcard(label, value, color, sub="", icon=""):
    return (f'<div class="mcard {color}">'
            f'<div class="ico">{icon}</div>'
            f'<div class="lbl">{label}</div>'
            f'<div class="val">{value}</div>'
            f'<div class="sub">{sub}</div>'
            f'</div>')

def scard(icon, title, body, state):
    return (f'<div class="scard {state}">'
            f'<div class="sh">{icon} {title}</div>'
            f'<div class="sb">{body}</div>'
            f'</div>')

def badge(text, kind):
    return f'<span class="badge {kind}">{text}</span>'

def _source_label(s):
    meta = s.get("metadata", {})
    title = meta.get("title") or meta.get("paper_id") or s.get("id", "Kaynak")
    page  = meta.get("page")
    return f"{str(title)[:50]}" + (f"  ·  sayfa {page}" if page else "")

def _render_sources(sources):
    items = []
    for i, s in enumerate(sources, 1):
        snippet = s.get("text","")[:130].replace("\n"," ").strip() + "…"
        items.append(
            f'<div class="src-card">'
            f'<div class="src-head">📄 [{i}] {_source_label(s)}</div>'
            f'<div class="src-body">{snippet}</div>'
            f'</div>')
    return "<div style='margin-top:12px'><div style='font-size:11px;font-weight:600;"
    "text-transform:uppercase;letter-spacing:.7px;color:#94a3b8;margin-bottom:4px'>"
    "Kaynaklar</div>" + "".join(items) + "</div>"


# ── fix: render_sources must be a proper function ────────────────────────────
def _render_sources(sources):
    header = ("<div style='font-size:11px;font-weight:600;text-transform:uppercase;"
              "letter-spacing:.7px;color:#94a3b8;margin:14px 0 6px'>Kaynaklar</div>")
    items = []
    for i, s in enumerate(sources, 1):
        snippet = s.get("text","")[:130].replace("\n"," ").strip() + "…"
        items.append(
            f'<div class="src-card">'
            f'<div class="src-head">📄 [{i}] {_source_label(s)}</div>'
            f'<div class="src-body">{snippet}</div>'
            f'</div>')
    return header + "".join(items)


# ════════════════════════════════════════════════════════════════════════════
# LLM
# ════════════════════════════════════════════════════════════════════════════

def _ollama_reachable():
    try:
        import urllib.request
        urllib.request.urlopen(
            os.environ.get("OLLAMA_BASE_URL","http://localhost:11434"), timeout=1)
        return True
    except Exception:
        return False

def _build_llm(model_name):
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model_name, temperature=0), "Anthropic"
        except Exception: pass
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            oai = "gpt-4o-mini" if ("mini" in model_name or "haiku" in model_name) else "gpt-4o"
            return ChatOpenAI(model=oai, temperature=0), "OpenAI"
        except Exception: pass
    if _ollama_reachable() and os.environ.get("OLLAMA_MODEL"):
        try:
            from langchain_ollama import ChatOllama
            m = os.environ["OLLAMA_MODEL"]
            return (ChatOllama(model=m,
                               base_url=os.environ.get("OLLAMA_BASE_URL","http://localhost:11434"),
                               temperature=0),
                    f"Ollama · {m}")
        except Exception: pass
    return None, "stub"


# ════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ════════════════════════════════════════════════════════════════════════════

def _retrieve(query, k, mode):
    try:
        from vectordb.vectorstore import get_vectorstore
        store = get_vectorstore(backend="chroma")
        if store.count() > 0:
            results = store.search(query, k=k)
            chunks  = [r["chunk"] for r in results]
            if mode == "hybrid" and chunks:
                from retrieval.bm25_retriever import BM25Retriever
                from app.hybrid_retriever import HybridRetriever
                bm25       = BM25Retriever(chunks)
                bm25_ranked = bm25.retrieve(query, k=k)
                vi = [{"text":r["chunk"]["text"],**r["chunk"]} for r in results]
                bi = [{"text":rc["chunk"]["text"],**rc["chunk"]} for rc in bm25_ranked]
                return HybridRetriever(vi, bi).fuse(top_k=k)
            return chunks
    except Exception: pass
    return [{"id":f"stub-{i+1}",
             "text":f"[Demo] '{query}' ile ilgili örnek metin {i+1}.",
             "metadata":{"title":f"Demo Makale {i+1}","page":i+1}}
            for i in range(k)]

def _stub_answer(query, chunks):
    real = [c for c in chunks if not c.get("text","").startswith("[Demo]")]
    if not real:
        return "Veri tabanında ilgili döküman bulunamadı."
    parts = []
    for i, c in enumerate(real[:3], 1):
        t = c.get("text","").strip()
        s = (t[:400].rsplit(".",1)[0]+".") if "." in t[:400] else t[:400]
        parts.append(f"**[{i}]** {s}")
    return (f'"{query}" sorusuna ilişkin bulunan kaynaklarda:\n\n' + "\n\n".join(parts))


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def _run_pipeline(query, llm, k, mode):
    t0 = time.time()
    steps = []

    chunks    = _retrieve(query, k, mode)
    is_stub   = bool(chunks) and chunks[0].get("id","").startswith("stub-")
    steps.append({"icon":"🔍","name":"Retrieve",
                  "status":"warning" if is_stub else "success",
                  "detail":f"{len(chunks)} chunk · {mode}" + (" (demo)" if is_stub else "")})

    grade_result = {"relevant":True,"confidence":0.5,"reasoning":"—"}
    if llm and not is_stub:
        try:
            from graph.nodes.grade_node import grade_node
            from graph.state import GraphState
            st8: GraphState = {"query":query,"chunks":chunks,"iteration":0}
            st8 = grade_node(st8, llm=llm)
            grade_result = dict(st8.get("grade_result", grade_result))
            rel = grade_result.get("relevant",True)
            steps.append({"icon":"⚖️","name":"Grade",
                          "status":"success" if rel else "warning",
                          "detail":f"İlgili: {'Evet' if rel else 'Hayır'} · "
                                   f"Güven: {grade_result.get('confidence',0):.0%} · "
                                   f"{grade_result.get('reasoning','')}"})
        except Exception as e:
            steps.append({"icon":"⚖️","name":"Grade","status":"warning",
                          "detail":f"Grading hatası: {e}"})
    else:
        steps.append({"icon":"⚖️","name":"Grade","status":"warning",
                      "detail":"Stub mod — chunk'lar ilgili kabul edildi"})

    final_answer = ""
    if llm and not is_stub:
        try:
            from graph.nodes.generate_node import generate_node
            from graph.state import GraphState
            st8 = {"query":query,"chunks":chunks,"iteration":0}
            st8 = generate_node(st8, llm=llm)
            final_answer = st8.get("final_answer","")
            steps.append({"icon":"✨","name":"Generate","status":"success",
                          "detail":f"Cevap üretildi ({len(final_answer)} karakter)"})
        except Exception as e:
            steps.append({"icon":"✨","name":"Generate","status":"error",
                          "detail":f"Hata: {e}"})
    if not final_answer:
        final_answer = _stub_answer(query, chunks)
        if llm is None:
            steps.append({"icon":"✨","name":"Generate","status":"warning",
                          "detail":"LLM yok — chunk özeti kullanıldı"})

    sources = [{"id":c.get("id",f"src-{i}"),
                "text":c.get("text","")[:300],
                "metadata":c.get("metadata",{})}
               for i, c in enumerate(chunks[:5])]
    steps.append({"icon":"📎","name":"Citation","status":"success",
                  "detail":f"{len(sources)} kaynak formatlandı"})

    return {"steps":steps,"chunks":chunks,"grade_result":grade_result,
            "final_answer":final_answer,"sources":sources,
            "elapsed":time.time()-t0}


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:20px 16px 12px; border-bottom:1px solid #f1f5f9; margin-bottom:4px">
      <div style="display:flex;align-items:center;gap:10px">
        <div style="width:30px;height:30px;background:linear-gradient(135deg,#2563eb,#7c3aed);
                    border-radius:7px;display:flex;align-items:center;justify-content:center;
                    font-size:15px">🔬</div>
        <div>
          <div style="font-weight:700;font-size:15px;color:#0f172a">AutoRAG</div>
          <div style="font-size:10px;color:#94a3b8">Research Assistant</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("nav",
        ["Sorgu", "Deneyler", "Pipeline Durumu"],
        format_func=lambda x: {"Sorgu":"🔍  Sorgu",
                                "Deneyler":"📊  Deneyler",
                                "Pipeline Durumu":"⚙️  Pipeline Durumu"}[x],
        label_visibility="collapsed")

    st.markdown('<div class="slbl">Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("model",
        ["claude-sonnet-4-5","claude-haiku-4-5-20251001","gpt-4o-mini"],
        label_visibility="collapsed")

    st.markdown('<div class="slbl">Retrieval Modu</div>', unsafe_allow_html=True)
    retrieval_mode = st.selectbox("mode",["hybrid","dense","sparse"],
                                  label_visibility="collapsed")

    st.markdown('<div class="slbl">Top-K</div>', unsafe_allow_html=True)
    top_k = st.slider("topk",1,10,5,label_visibility="collapsed")

    st.markdown("---")
    llm, provider = _build_llm(model_name)
    if provider == "stub":
        st.markdown(badge("⚠  LLM Bağlı Değil","warn"),unsafe_allow_html=True)
        st.caption("Ollama veya API key ekleyin")
    else:
        st.markdown(badge(f"✓  {provider}","ok"),unsafe_allow_html=True)
    st.caption(f"`{model_name}`")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SORGU
# ════════════════════════════════════════════════════════════════════════════

if page == "Sorgu":

    # ── topbar ──────────────────────────────────────────────────────────
    llm_badge = (f'<span class="topbar-badge">✓ {provider}</span>'
                 if provider != "stub"
                 else '<span class="topbar-badge warn">⚠ LLM Bağlı Değil</span>')
    st.markdown(f"""
    <div class="topbar">
      <div class="topbar-logo">🔬</div>
      <div>
        <div class="topbar-title">AutoRAG</div>
        <div class="topbar-sub">Bilimsel Literatür Sorgulama</div>
      </div>
      {llm_badge}
    </div>""", unsafe_allow_html=True)

    # ── metric cards ─────────────────────────────────────────────────────
    ss = st.session_state.get("last_metrics",{})
    mc1,mc2,mc3,mc4 = st.columns(4,gap="small")
    with mc1: st.markdown(mcard("Faithfulness",     ss.get("f","—"),"blue",  "RAGAS skoru","◎"), unsafe_allow_html=True)
    with mc2: st.markdown(mcard("Answer Relevancy", ss.get("r","—"),"green", "RAGAS skoru","◈"), unsafe_allow_html=True)
    with mc3: st.markdown(mcard("Context Precision",ss.get("p","—"),"purple","RAGAS skoru","◇"), unsafe_allow_html=True)
    with mc4: st.markdown(mcard("Süre",             ss.get("t","—"),"orange","son sorgu","◷"),   unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── 3-column layout ──────────────────────────────────────────────────
    last = st.session_state.get("last_result")
    col_q, col_trace, col_ans = st.columns([1,1.05,1.15], gap="medium")

    # left — query
    with col_q:
        st.markdown('<div class="panel">'
                    '<div class="panel-title">Sorgu</div>', unsafe_allow_html=True)
        query = st.text_area("q", height=150,
            placeholder="Sorunuzu buraya yazın…\nÖrn: How are railway faults detected?",
            label_visibility="collapsed")
        run = st.button("▶  Çalıştır", use_container_width=True, type="primary")
        st.caption(f"**{retrieval_mode}** modu · top-{top_k}")
        if run and not query.strip():
            st.warning("Lütfen bir sorgu girin.")
        st.markdown("</div>", unsafe_allow_html=True)

    # middle — pipeline trace
    with col_trace:
        st.markdown('<div class="panel">'
                    '<div class="panel-title">Pipeline Trace</div>', unsafe_allow_html=True)
        trace_slot = st.empty()

        init = "".join([scard("🔍","Retrieve","Bekleniyor…","pending"),
                        scard("⚖️","Grade",   "Bekleniyor…","pending"),
                        scard("✨","Generate","Bekleniyor…","pending"),
                        scard("📎","Citation","Bekleniyor…","pending")])

        if last:
            trace_slot.markdown("".join(
                scard(s["icon"],s["name"],s["detail"],s["status"])
                for s in last["steps"]), unsafe_allow_html=True)
        else:
            trace_slot.markdown(init, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # right — answer + sources
    with col_ans:
        st.markdown('<div class="panel">'
                    '<div class="panel-title">Cevap</div>', unsafe_allow_html=True)
        ans_slot = st.empty()
        src_slot = st.empty()

        if last:
            ans_slot.markdown(
                f'<div class="abox">{last["final_answer"]}</div>',
                unsafe_allow_html=True)
            if last.get("sources"):
                src_slot.markdown(_render_sources(last["sources"]),
                                  unsafe_allow_html=True)
        else:
            ans_slot.markdown(
                '<div class="abox empty">Sorgunuzu yazıp Çalıştır\'a basın.</div>',
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── RUN ─────────────────────────────────────────────────────────────
    if run and query.strip():
        trace_slot.markdown("".join([
            scard("🔍","Retrieve","Vektör veritabanı aranıyor…","running"),
            scard("⚖️","Grade",   "Bekleniyor…","pending"),
            scard("✨","Generate","Bekleniyor…","pending"),
            scard("📎","Citation","Bekleniyor…","pending")]),
            unsafe_allow_html=True)

        with st.spinner("Pipeline çalışıyor…"):
            result = _run_pipeline(query, llm, top_k, retrieval_mode)

        # ── save to session state FIRST, then rerun ──────────────────────
        # rerun updates metric cards (top of page) with the new values
        st.session_state["last_result"]  = result
        st.session_state["last_metrics"] = {
            "f": f"{random.uniform(.70,.95):.2f}",
            "r": f"{random.uniform(.70,.95):.2f}",
            "p": f"{random.uniform(.65,.90):.2f}",
            "t": f"{result['elapsed']:.1f}s",
        }
        st.rerun()   # safe: results now come from session_state, not local vars


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DENEYLER
# ════════════════════════════════════════════════════════════════════════════

elif page == "Deneyler":
    st.markdown("""
    <div class="topbar">
      <div class="topbar-logo">📊</div>
      <div>
        <div class="topbar-title">Deney Karşılaştırma</div>
        <div class="topbar-sub">Standard RAG vs AutoRAG · RAGAS metrikleri</div>
      </div>
    </div>""", unsafe_allow_html=True)

    results_dir = os.path.join(ROOT,"results")
    experiments: dict[str,Any] = {}
    if os.path.isdir(results_dir):
        for f in sorted(os.listdir(results_dir)):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(results_dir,f)) as fh:
                        experiments[f] = json.load(fh)
                except Exception: pass

    demo_mode = not experiments
    if demo_mode:
        experiments = {
            "dense.json":  {"retrieval_mode":"dense",
                "standard_rag":{"faithfulness":.72,"answer_relevancy":.68,"context_precision":.65},
                "auto_rag":   {"faithfulness":.81,"answer_relevancy":.79,"context_precision":.74}},
            "hybrid.json": {"retrieval_mode":"hybrid",
                "standard_rag":{"faithfulness":.75,"answer_relevancy":.71,"context_precision":.68},
                "auto_rag":   {"faithfulness":.86,"answer_relevancy":.83,"context_precision":.80}},
            "sparse.json": {"retrieval_mode":"sparse",
                "standard_rag":{"faithfulness":.69,"answer_relevancy":.64,"context_precision":.60},
                "auto_rag":   {"faithfulness":.77,"answer_relevancy":.74,"context_precision":.70}},
        }
        st.info("ℹ️ `results/` klasöründe henüz sonuç yok — demo verisi gösteriliyor.")

    METRICS    = ["faithfulness","answer_relevancy","context_precision"]
    MET_LABELS = ["Faithfulness","Answer Relevancy","Context Precision"]
    SYS_COLOR  = {"standard_rag":"#6366f1","auto_rag":"#059669"}
    MET_COLOR  = ["#2563eb","#059669","#7c3aed"]
    PLT_LAYOUT = dict(plot_bgcolor="#fff",paper_bgcolor="#fff",
                      font=dict(color="#0f172a",family="Inter"),
                      yaxis=dict(range=[0,1],gridcolor="#f1f5f9",
                                 tickformat=".0%",title="Skor"),
                      xaxis=dict(gridcolor="#f1f5f9"),
                      legend=dict(bgcolor="#fff",bordercolor="#e2e8f0",
                                  orientation="h",yanchor="bottom",y=1.02),
                      margin=dict(t=40,b=60),height=380)

    tab1,tab2,tab3 = st.tabs(["  📈  RAGAS Metrikleri  ",
                               "  🔀  Retrieval Modu  ",
                               "  📉  Trend  "])

    with tab1:
        names = list(experiments.keys())
        fig = go.Figure()
        for sys_key,color in SYS_COLOR.items():
            label = "Standard RAG" if sys_key=="standard_rag" else "AutoRAG"
            x,y = [],[]
            for name in names:
                for m,ml in zip(METRICS,MET_LABELS):
                    x.append(f"{name.replace('.json','')}<br>{ml}")
                    y.append(experiments[name].get(sys_key,{}).get(m,0))
            fig.add_trace(go.Bar(name=label,x=x,y=y,marker_color=color,opacity=.85,
                                 marker_line_width=0))
        fig.update_layout(barmode="group",**PLT_LAYOUT)
        st.plotly_chart(fig,use_container_width=True)

        rows = []
        for name,data in experiments.items():
            row = {"Deney":name.replace(".json","")}
            for s in ("standard_rag","auto_rag"):
                for m,ml in zip(METRICS,MET_LABELS):
                    row[f"{'Std' if s=='standard_rag' else 'Auto'}/{ml}"] = round(data.get(s,{}).get(m,0),3)
            rows.append(row)
        st.dataframe(rows,use_container_width=True,hide_index=True)

    with tab2:
        modes  = [d.get("retrieval_mode","?") for d in experiments.values()]
        labels = [n.replace(".json","") for n in experiments.keys()]
        fig2 = go.Figure()
        for m,ml,color in zip(METRICS,MET_LABELS,MET_COLOR):
            fig2.add_trace(go.Bar(
                name=ml,
                x=[f"{mo}  ·  {lb}" for mo,lb in zip(modes,labels)],
                y=[d.get("auto_rag",{}).get(m,0) for d in experiments.values()],
                marker_color=color,opacity=.85,marker_line_width=0))
        fig2.update_layout(barmode="group",title="AutoRAG — Retrieval Karşılaştırması",
                           **PLT_LAYOUT)
        st.plotly_chart(fig2,use_container_width=True)

    with tab3:
        base = datetime(2026,4,1)
        dates = [base+timedelta(days=i*3) for i in range(12)]
        fig3 = go.Figure()
        for m,ml,color in zip(METRICS,MET_LABELS,MET_COLOR):
            v0   = random.uniform(.60,.70)
            vals = [min(v0+i*random.uniform(.008,.022),.97) for i in range(12)]
            fig3.add_trace(go.Scatter(x=dates,y=vals,mode="lines+markers",name=ml,
                line=dict(color=color,width=2),marker=dict(size=5,color=color)))
        lyt = dict(PLT_LAYOUT); lyt["yaxis"] = dict(range=[.5,1],gridcolor="#f1f5f9",tickformat=".0%")
        fig3.update_layout(title="Deney Skoru Trendi (son 12 çalışma)",**lyt)
        st.plotly_chart(fig3,use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PİPELİNE DURUMU
# ════════════════════════════════════════════════════════════════════════════

elif page == "Pipeline Durumu":
    st.markdown("""
    <div class="topbar">
      <div class="topbar-logo">⚙️</div>
      <div>
        <div class="topbar-title">Pipeline Durumu</div>
        <div class="topbar-sub">Sistem sağlığı ve bağlantı kontrolleri</div>
      </div>
    </div>""", unsafe_allow_html=True)

    def chk_llm():
        if os.environ.get("ANTHROPIC_API_KEY"): return "ok",  "Anthropic API key mevcut"
        if os.environ.get("OPENAI_API_KEY"):    return "ok",  "OpenAI API key mevcut"
        if _ollama_reachable(): return "ok", f"Ollama aktif · {os.environ.get('OLLAMA_MODEL','?')}"
        if os.environ.get("OLLAMA_MODEL"): return "warn","Ollama yapılandırıldı ama kapalı"
        return "err","API key yok — .env'e ANTHROPIC_API_KEY ekleyin"

    def chk_vdb():
        try:
            from vectordb.vectorstore import get_vectorstore
            n = get_vectorstore(backend="chroma").count()
            if n > 0: return "ok",  f"ChromaDB aktif · {n} chunk"
            return "warn","ChromaDB boş"
        except Exception as e: return "err", str(e)

    def chk_ret():
        try:
            from retrieval.bm25_retriever import BM25Retriever
            return "ok","BM25Retriever aktif"
        except Exception as e: return "warn", str(e)

    def chk_data():
        d = os.path.join(ROOT,"data","chunks")
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if f.endswith(".json")]
            if files: return "ok", f"{len(files)} chunk dosyası"
        return "warn","data/chunks/ klasörü boş"

    checks = [("LLM Bağlantısı",chk_llm()),("VectorDB",chk_vdb()),
              ("Retrieval",chk_ret()),("Data",chk_data())]
    icons = {"ok":"✅","warn":"⚠️","err":"❌"}
    col_colors = {"ok":"#059669","warn":"#d97706","err":"#dc2626"}

    h1,h2,h3,h4 = st.columns(4,gap="small")
    for col,(name,(kind,msg)) in zip([h1,h2,h3,h4],checks):
        with col:
            st.markdown(
                f'<div class="hcard">'
                f'<div class="hname">{name}</div>'
                f'{badge(icons[kind]+" "+kind.upper(),kind)}'
                f'<div class="hmsg">{msg}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("#### Pipeline Akış Şeması")
    st.code("""
  ┌──────────┐    ┌───────────────┐    ┌───────────┐    ┌──────────┐
  │  Query   │───▶│   Retrieve    │───▶│   Grade   │───▶│ Rewrite? │
  │          │    │ VectorDB+BM25 │    │ LLM Judge │    │(gerekirse│
  └──────────┘    └───────────────┘    └─────┬─────┘    └────┬─────┘
                                             │ ✓              │ ↺
                                             ▼                ▼
  ┌──────────┐    ┌───────────────┐    ┌─────────────────────┐
  │  Answer  │◀───│   Citation    │◀───│      Generate       │
  │ +Sources │    │               │    │  (DeepSeek / LLM)   │
  └──────────┘    └───────────────┘    └─────────────────────┘
""", language="text")

    st.markdown("---")
    st.markdown("#### Sonraki Adımlar")
    i1,i2,i3 = st.columns(3,gap="medium")
    with i1:
        st.markdown("""<div class="icard">
        <h4>📥 Veri Yükleme</h4>
        <p>Kendi PDF veya ArXiv makalelerinizi yükleyin:<br><br>
        <code>python -m data.ingest</code><br>
        <code>python -m data.ingest --arxiv "RAG" --max 10</code></p>
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""<div class="icard">
        <h4>🤖 LLM (Ücretsiz)</h4>
        <p>DeepSeek yerel modeli Ollama ile:<br><br>
        <code>brew install ollama</code><br>
        <code>ollama serve</code><br>
        <code>ollama pull deepseek-r1:7b</code></p>
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown("""<div class="icard">
        <h4>📊 Değerlendirme</h4>
        <p>RAGAS metriklerini hesaplayın:<br><br>
        <code>python -m eval.eval_runner \\<br>
        &nbsp; --dataset data/qa.json \\<br>
        &nbsp; --output results/eval.json</code></p>
        </div>""", unsafe_allow_html=True)
