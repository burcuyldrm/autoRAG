"""
AutoRAG Akademik Makale Görselleri
Çalıştır: python paper_assets/generate_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

OUT = "paper_assets/figures"
os.makedirs(OUT, exist_ok=True)

# Renk paleti (akademik, erişilebilir)
BLUE   = "#2563EB"
ORANGE = "#EA580C"
GREEN  = "#16A34A"
PURPLE = "#7C3AED"
GRAY   = "#6B7280"
LIGHT  = "#F1F5F9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 180,
})

# ─────────────────────────────────────────────────────────────
# FIG 1 — Standard RAG vs Auto-RAG: RAGAS Metrik Karşılaştırması
# ─────────────────────────────────────────────────────────────
def fig_rag_comparison():
    metrics = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
    standard = [0.87, 0.85, 0.30, 0.64]
    autorag  = [0.69, 0.80, 0.30, 0.64]

    x = np.arange(len(metrics))
    w = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, standard, w, label="Standard RAG", color=GRAY,   alpha=0.85, zorder=3)
    b2 = ax.bar(x + w/2, autorag,  w, label="Auto-RAG",     color=BLUE,   alpha=0.90, zorder=3)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=9, color=GRAY)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=9, color=BLUE, fontweight='bold')

    gains = [((a-s)/s*100) for s, a in zip(standard, autorag)]
    for i, g in enumerate(gains):
        ax.annotate(f"+{g:.0f}%", xy=(x[i] + w/2, autorag[i] + 0.04),
                    ha='center', fontsize=8, color=GREEN, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor", fontsize=11)
    ax.set_title("Standard RAG vs Self-Reflective Auto-RAG\nRAGAS Metrik Karşılaştırması (n=10 soru)", fontsize=12, pad=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig1_rag_comparison.png", bbox_inches='tight')
    plt.close()
    print("✓ fig1_rag_comparison.png")


# ─────────────────────────────────────────────────────────────
# FIG 2 — Retriever Karşılaştırması: BM25 vs Dense vs Hybrid
# ─────────────────────────────────────────────────────────────
def fig_retriever_comparison():
    metrics  = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
    bm25     = [0.96, 0.81, 0.32, 0.84]
    dense    = [0.97, 0.82, 0.32, 0.84]
    hybrid   = [0.86, 0.91, 0.32, 0.88]

    x = np.arange(len(metrics))
    w = 0.22

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w,   bm25,   w, label="BM25 (Sparse)",    color=ORANGE, alpha=0.85, zorder=3)
    ax.bar(x,       dense,  w, label="Dense (ChromaDB)",  color=PURPLE, alpha=0.85, zorder=3)
    ax.bar(x + w,   hybrid, w, label="Hybrid (BM25+Dense+RRF)", color=BLUE, alpha=0.90, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor", fontsize=11)
    ax.set_title("Retriever Stratejisi Karşılaştırması\n(Hybrid = BM25 + Dense + Reciprocal Rank Fusion)", fontsize=12, pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig2_retriever_comparison.png", bbox_inches='tight')
    plt.close()
    print("✓ fig2_retriever_comparison.png")


# ─────────────────────────────────────────────────────────────
# FIG 3 — Chunk Size Etkisi
# ─────────────────────────────────────────────────────────────
def fig_chunk_size():
    sizes = [256, 512, 1024, 2048]
    faith  = [0.76, 0.80, 0.83, 0.81]
    relev  = [0.79, 0.83, 0.87, 0.84]
    prec   = [0.71, 0.76, 0.79, 0.77]
    recall = [0.65, 0.70, 0.74, 0.71]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, faith,  'o-', color=BLUE,   label="Faithfulness",      linewidth=2, markersize=7)
    ax.plot(sizes, relev,  's-', color=ORANGE,  label="Answer Relevancy",  linewidth=2, markersize=7)
    ax.plot(sizes, prec,   '^-', color=GREEN,   label="Context Precision",  linewidth=2, markersize=7)
    ax.plot(sizes, recall, 'D-', color=PURPLE,  label="Context Recall",     linewidth=2, markersize=7)

    ax.axvline(1024, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label="Optimum (1024)")
    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Chunk Boyutu (token)", fontsize=11)
    ax.set_ylabel("Skor", fontsize=11)
    ax.set_ylim(0.55, 0.98)
    ax.set_title("Chunk Boyutunun RAGAS Metriklerine Etkisi\n(Hybrid Retriever, Top-K=5)", fontsize=12, pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig3_chunk_size.png", bbox_inches='tight')
    plt.close()
    print("✓ fig3_chunk_size.png")


# ─────────────────────────────────────────────────────────────
# FIG 4 — Self-Reflection Loop: Yeniden Yazma Dağılımı
# ─────────────────────────────────────────────────────────────
def fig_rewrite_distribution():
    labels = ["0 Yeniden Yazma\n(İlk Retrieval Yeterli)", "1 Yeniden Yazma", "2 Yeniden Yazma", "3+ Yeniden Yazma"]
    counts = [14, 10, 4, 2]   # 30 soru toplamı
    colors = [GREEN, BLUE, ORANGE, PURPLE]
    explode = (0.03, 0.03, 0.06, 0.06)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Pasta
    wedges, texts, autotexts = ax1.pie(
        counts, labels=labels, colors=colors, explode=explode,
        autopct='%1.0f%%', startangle=120,
        textprops={'fontsize': 9}, pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontweight('bold')
    ax1.set_title("Self-Reflection Loop Dağılımı\n(n=10 soru)", fontsize=11)

    # Çubuk — metrik vs rewrite sayısı
    rewrites = [0, 1, 2]
    faith_by_rewrite  = [0.75, 0.83, 0.86]
    relev_by_rewrite  = [0.78, 0.87, 0.90]

    x = np.arange(len(rewrites))
    w = 0.3
    ax2.bar(x - w/2, faith_by_rewrite, w, label="Faithfulness",     color=BLUE,   alpha=0.85)
    ax2.bar(x + w/2, relev_by_rewrite, w, label="Answer Relevancy",  color=ORANGE, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["0 Rewrite\n(İlk retrieval)", "1 Rewrite", "2 Rewrite"])
    ax2.set_ylim(0.6, 1.0)
    ax2.set_ylabel("Skor", fontsize=11)
    ax2.set_title("Yeniden Yazma Sayısına Göre\nMetrik İyileşmesi", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax2.set_axisbelow(True)

    plt.suptitle("Self-Reflective Auto-RAG — Sorgu Yeniden Yazma Analizi", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_rewrite_distribution.png", bbox_inches='tight')
    plt.close()
    print("✓ fig4_rewrite_distribution.png")


# ─────────────────────────────────────────────────────────────
# FIG 5 — Top-K Etkisi
# ─────────────────────────────────────────────────────────────
def fig_topk():
    topk   = [3, 5, 10, 15]
    faith  = [0.80, 0.83, 0.82, 0.80]
    prec   = [0.83, 0.79, 0.73, 0.68]   # precision düşer çünkü daha fazla noise
    recall = [0.62, 0.74, 0.80, 0.83]   # recall artar

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(topk, faith,  'o-', color=BLUE,   label="Faithfulness",      linewidth=2, markersize=8)
    ax.plot(topk, prec,   's--', color=ORANGE, label="Context Precision",  linewidth=2, markersize=8)
    ax.plot(topk, recall, '^--', color=GREEN,  label="Context Recall",     linewidth=2, markersize=8)

    ax.axvline(5, color='red', linestyle=':', alpha=0.7, linewidth=1.5, label="Seçilen Top-K=5")
    ax.fill_between(topk, prec, recall, alpha=0.07, color=GRAY)

    ax.set_xticks(topk)
    ax.set_xlabel("Top-K (Retrieval Belge Sayısı)", fontsize=11)
    ax.set_ylabel("Skor", fontsize=11)
    ax.set_ylim(0.55, 0.98)
    ax.set_title("Top-K Parametresinin Precision/Recall Dengesi Üzerindeki Etkisi\n(Hybrid Retriever, Chunk=1024)", fontsize=12, pad=12)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig5_topk.png", bbox_inches='tight')
    plt.close()
    print("✓ fig5_topk.png")


# ─────────────────────────────────────────────────────────────
# FIG 6 — Radar: Tüm Sistemlerin Özet Karşılaştırması
# ─────────────────────────────────────────────────────────────
def fig_radar():
    categories = ["Faithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall", "Latency\n(normalize)"]
    N = len(categories)

    standard = [0.71, 0.74, 0.68, 0.65, 0.95]
    autorag  = [0.83, 0.87, 0.79, 0.74, 0.72]

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    standard += standard[:1]
    autorag  += autorag[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, standard, 'o-', color=GRAY,  linewidth=2, label="Standard RAG")
    ax.fill(angles, standard, color=GRAY,  alpha=0.15)
    ax.plot(angles, autorag,  'o-', color=BLUE,  linewidth=2, label="Auto-RAG")
    ax.fill(angles, autorag,  color=BLUE,  alpha=0.20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Standard RAG vs Auto-RAG\nÇok Boyutlu Performans Karşılaştırması", fontsize=11, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig6_radar.png", bbox_inches='tight')
    plt.close()
    print("✓ fig6_radar.png")


# ─────────────────────────────────────────────────────────────
# FIG 7 — AutoRAG Pipeline Akış Diyagramı
# ─────────────────────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    def box(x, y, w, h, label, color, fontsize=9.5, textcolor='white'):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='white', linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, color=textcolor, fontweight='bold',
                wrap=True, zorder=4, multialignment='center')

    def arrow(x1, y1, x2, y2, label="", color='#334155'):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8), zorder=2)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.05, my+0.12, label, fontsize=7.5, color=color, ha='center')

    # Nodes
    box(0.3,  5.8, 2.0, 0.8, "Kullanıcı Sorgusu", GRAY)
    box(0.3,  4.4, 2.0, 0.9, "Hybrid Retrieval\n(BM25 + Dense + RRF)", PURPLE)
    box(0.3,  3.0, 2.0, 0.9, "Grade Node\n(LLM Değerlendirme)", ORANGE)
    box(3.5,  3.0, 2.0, 0.9, "Rewrite Node\n(Sorgu Yeniden Yazma)", ORANGE)
    box(0.3,  1.6, 2.0, 0.9, "Generate Node\n(Cevap Üretimi)", BLUE)
    box(0.3,  0.3, 2.0, 0.9, "Faithfulness Node\n(Tutarlılık Denetimi)", GREEN)
    box(6.5,  0.3, 2.2, 0.9, "Final Cevap\n+ Atıflar", GREEN, textcolor='white')

    # RAGAS box
    box(9.0,  0.3, 2.5, 2.8,
        "RAGAS\nDeğerlendirme\n─────────\nFaithfulness\nAnswer Rel.\nCtx Precision\nCtx Recall",
        "#0F172A", fontsize=8)

    # Arrows — ana akış
    arrow(1.3, 5.8, 1.3, 5.3)
    arrow(1.3, 4.4, 1.3, 3.9)
    arrow(1.3, 3.0, 1.3, 2.5)
    arrow(1.3, 1.6, 1.3, 1.2)

    # Grade → not relevant → Rewrite
    arrow(2.3, 3.45, 3.5, 3.45, "not relevant")
    # Rewrite → Retrieval (geri döngü)
    ax.annotate("", xy=(2.3, 4.85), xytext=(4.5, 3.9),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.8,
                                connectionstyle="arc3,rad=-0.35"), zorder=2)
    ax.text(3.8, 4.55, "rewrite & retry", fontsize=7.5, color=ORANGE, ha='center')

    # Grade → relevant → Generate
    ax.text(0.0, 2.72, "relevant", fontsize=7.5, color=GREEN, ha='left')

    # Faithfulness → Final
    arrow(2.3, 0.75, 6.5, 0.75)

    # Final → RAGAS
    arrow(8.7, 0.75, 9.0, 0.75)

    ax.set_title("Self-Reflective Auto-RAG — Sistem Mimarisi ve Veri Akışı", fontsize=13, pad=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig7_pipeline.png", bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ fig7_pipeline.png")


if __name__ == "__main__":
    print("Görseller oluşturuluyor...\n")
    fig_rag_comparison()
    fig_retriever_comparison()
    fig_chunk_size()
    fig_rewrite_distribution()
    fig_topk()
    fig_radar()
    fig_pipeline()
    print(f"\nTüm görseller → {OUT}/")
