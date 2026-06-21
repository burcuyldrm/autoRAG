"""
Update paper results tables and figures with real benchmark data.
Run after eval/run_full_benchmark.py has completed.

Usage:
    python paper_assets/update_paper_results.py
"""
from __future__ import annotations

import json
import os
import re
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS = os.path.join(ROOT, "results")
PAPER_TXT = os.path.join(ROOT, "paper_assets", "makale_autorag.txt")
FIGURES_PY = os.path.join(ROOT, "paper_assets", "generate_figures.py")


def load_rag_comparison():
    path = os.path.join(RESULTS, "benchmark_rag_comparison.json")
    with open(path) as f:
        data = json.load(f)
    exps = {e["experiment_name"]: e for e in data["experiments"]}
    std = exps["standard_rag"]
    auto = exps["auto_rag"]
    return std, auto


def load_retriever():
    path = os.path.join(RESULTS, "benchmark_retriever.json")
    with open(path) as f:
        data = json.load(f)
    return {e["retriever_type"]: e for e in data["experiments"]}


def load_topk():
    path = os.path.join(RESULTS, "benchmark_topk.json")
    with open(path) as f:
        data = json.load(f)
    return {e["top_k"]: e for e in data["experiments"]}


def pct_change(old, new):
    if old and old != 0:
        return (new - old) / old * 100
    return 0.0


def update_paper_table_iv(std, auto):
    """Update Tablo IV — Standard RAG vs Self-Reflective Auto-RAG"""
    f_std = std.get("faithfulness", 0)
    ar_std = std.get("answer_relevancy", 0)
    cp_std = std.get("context_precision", 0)
    cr_std = std.get("context_recall", 0)
    lat_std = std.get("avg_latency_seconds", 0)

    f_auto = auto.get("faithfulness", 0)
    ar_auto = auto.get("answer_relevancy", 0)
    cp_auto = auto.get("context_precision", 0)
    cr_auto = auto.get("context_recall", 0)
    lat_auto = auto.get("avg_latency_seconds", 0)
    rew_auto = auto.get("avg_rewrite_count", 0)

    n_q = std.get("n_questions", 10)

    table = f"""
  Tablo IV — Standard RAG vs Self-Reflective Auto-RAG (n={n_q} soru)
  ──────────────────────────────────────────────────────────────────────────
  Metrik              | Standard RAG | Self-Refl. Auto-RAG | Δ (%)
  ──────────────────────────────────────────────────────────────────────────
  Faithfulness        |    {f_std:.2f}      |       {f_auto:.2f}          | {pct_change(f_std,f_auto):+.1f}%
  Answer Relevancy    |    {ar_std:.2f}      |       {ar_auto:.2f}          | {pct_change(ar_std,ar_auto):+.1f}%
  Context Precision   |    {cp_std:.2f}      |       {cp_auto:.2f}          | {pct_change(cp_std,cp_auto):+.1f}%
  Context Recall      |    {cr_std:.2f}      |       {cr_auto:.2f}          | {pct_change(cr_std,cr_auto):+.1f}%
  ──────────────────────────────────────────────────────────────────────────
  Ort. Latency (sn)   |    {lat_std:.2f}      |       {lat_auto:.2f}          | {pct_change(lat_std,lat_auto):+.0f}%
  Ort. Rewrite Sayısı |    0.0       |       {rew_auto:.2f}          | —
  ──────────────────────────────────────────────────────────────────────────"""
    return table, f_std, ar_std, cp_std, cr_std, f_auto, ar_auto, cp_auto, cr_auto


def update_paper_table_v(ret):
    bm25 = ret.get("bm25", {})
    dense = ret.get("dense", {})
    hybrid = ret.get("hybrid", {})
    n_q = bm25.get("n_questions", 5)

    def row(r):
        return (r.get("faithfulness",0), r.get("answer_relevancy",0),
                r.get("context_precision",0), r.get("context_recall",0))

    fb,ab,pb,rb = row(bm25)
    fd,ad,pd,rd = row(dense)
    fh,ah,ph,rh = row(hybrid)

    table = f"""
  Tablo V — Retriever Karşılaştırması (n={n_q} soru, chunk=1024, Top-K=5)
  ──────────────────────────────────────────────────────────────────────────
  Retriever     | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall
  ──────────────────────────────────────────────────────────────────────────
  BM25          |    {fb:.2f}      |    {ab:.2f}     |    {pb:.2f}       |   {rb:.2f}
  Dense         |    {fd:.2f}      |    {ad:.2f}     |    {pd:.2f}       |   {rd:.2f}
  Hybrid (RRF)  |    {fh:.2f}      |    {ah:.2f}     |    {ph:.2f}       |   {rh:.2f}
  ──────────────────────────────────────────────────────────────────────────"""
    return table, (fb,fd,fh), (ab,ad,ah), (pb,pd,ph), (rb,rd,rh)


def update_paper_table_vii(topk):
    n_q = list(topk.values())[0].get("n_questions", 5) if topk else 5
    rows = []
    for k in [3, 5, 10, 15]:
        e = topk.get(k, {})
        f_ = e.get("faithfulness", 0)
        cp = e.get("context_precision", 0)
        cr = e.get("context_recall", 0)
        opt = " ← seçilen" if k == 5 else ""
        rows.append(f"  {k:2d}   |    {f_:.2f}      |    {cp:.2f}       |   {cr:.2f}{opt}")
    rows_str = "\n".join(rows)
    table = f"""
  Tablo VII — Top-K Analizi (Hybrid Retriever, chunk=1024, n={n_q}/koşul)
  ──────────────────────────────────────────────────────────────────────────
  Top-K | Faithfulness | Ctx Precision | Ctx Recall
  ──────────────────────────────────────────────────────────────────────────
{rows_str}
  ──────────────────────────────────────────────────────────────────────────"""
    return table


def update_figures_py(f_std, ar_std, cp_std, cr_std, f_auto, ar_auto, cp_auto, cr_auto,
                       fb, fd, fh, ab, ad, ah, pb, pd, ph, rb, rd, rh):
    """Update hardcoded values in generate_figures.py"""
    with open(FIGURES_PY) as fp:
        code = fp.read()

    # Update fig_rag_comparison
    code = re.sub(
        r'(def fig_rag_comparison\(\):.*?standard = )\[[\d., ]+\]',
        rf'\1[{f_std:.2f}, {ar_std:.2f}, {cp_std:.2f}, {cr_std:.2f}]',
        code, flags=re.DOTALL, count=1
    )
    code = re.sub(
        r'(def fig_rag_comparison\(\):.*?autorag\s*=\s*)\[[\d., ]+\]',
        rf'\1[{f_auto:.2f}, {ar_auto:.2f}, {cp_auto:.2f}, {cr_auto:.2f}]',
        code, flags=re.DOTALL, count=1
    )

    # Update fig_retriever_comparison
    code = re.sub(
        r'(def fig_retriever_comparison\(\):.*?bm25\s*=\s*)\[[\d., ]+\]',
        rf'\1[{fb:.2f}, {ab:.2f}, {pb:.2f}, {rb:.2f}]',
        code, flags=re.DOTALL, count=1
    )
    code = re.sub(
        r'(def fig_retriever_comparison\(\):.*?dense\s*=\s*)\[[\d., ]+\]',
        rf'\1[{fd:.2f}, {ad:.2f}, {pd:.2f}, {rd:.2f}]',
        code, flags=re.DOTALL, count=1
    )
    code = re.sub(
        r'(def fig_retriever_comparison\(\):.*?hybrid\s*=\s*)\[[\d., ]+\]',
        rf'\1[{fh:.2f}, {ah:.2f}, {ph:.2f}, {rh:.2f}]',
        code, flags=re.DOTALL, count=1
    )

    # Update title n_questions
    code = re.sub(r'n=30 soru', 'n=10 soru', code)

    with open(FIGURES_PY, "w") as fp:
        fp.write(code)
    print("✓ generate_figures.py updated with real benchmark values")


def main():
    print("Loading benchmark results...")
    try:
        std, auto = load_rag_comparison()
    except FileNotFoundError:
        print("ERROR: benchmark_rag_comparison.json not found. Run benchmarks first.")
        sys.exit(1)

    try:
        ret = load_retriever()
    except FileNotFoundError:
        print("WARNING: benchmark_retriever.json not found, skipping retriever table")
        ret = {}

    try:
        topk = load_topk()
        topk_by_k = {e["top_k"]: e for e in topk.get("experiments", [])} if isinstance(topk, dict) and "experiments" in topk else {}
    except Exception:
        topk_by_k = {}

    print("\n=== Standard RAG ===")
    print(f"  F={std.get('faithfulness',0):.3f} AR={std.get('answer_relevancy',0):.3f} "
          f"CP={std.get('context_precision',0):.3f} CR={std.get('context_recall',0):.3f}")

    print("\n=== Auto-RAG ===")
    print(f"  F={auto.get('faithfulness',0):.3f} AR={auto.get('answer_relevancy',0):.3f} "
          f"CP={auto.get('context_precision',0):.3f} CR={auto.get('context_recall',0):.3f}")

    table_iv, f_std, ar_std, cp_std, cr_std, f_auto, ar_auto, cp_auto, cr_auto = update_paper_table_iv(std, auto)
    print("\nTablo IV:")
    print(table_iv)

    if ret:
        table_v, (fb,fd,fh), (ab,ad,ah), (pb,pd,ph), (rb,rd,rh) = update_paper_table_v(ret)
        print("\nTablo V:")
        print(table_v)

        if topk_by_k:
            table_vii = update_paper_table_vii(topk_by_k)
            print("\nTablo VII:")
            print(table_vii)

        # Update figures
        update_figures_py(f_std, ar_std, cp_std, cr_std, f_auto, ar_auto, cp_auto, cr_auto,
                          fb, fd, fh, ab, ad, ah, pb, pd, ph, rb, rd, rh)

    print("\n✓ Paper results computed. Copy Tablo IV–VII into the paper text manually.")
    print("  Run: python paper_assets/generate_figures.py  — to regenerate figures")


if __name__ == "__main__":
    main()
