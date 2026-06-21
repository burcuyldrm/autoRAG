# Rewrite Mekanizması: Analiz ve Deneysel Tasarım

## 1. İlk Benchmark'ta Rewrite Neden Tetiklenmedi?

İlk benchmark (amnesty_qa, `grade_threshold=0.60`) tüm sorularda `avg_rewrite_count=0.00` verdi. İki kök neden vardır:

### 1.1 Grade Prompt'unda Hardcoded Örnek Değer

`grade_node.py`'daki orijinal prompt şunu içeriyordu:

```
Reply with ONLY a JSON object:
{"relevant": true, "confidence": 0.85, "reasoning": "one sentence"}
```

qwen2.5:3b, prompt'taki `0.85` değerini pattern-match yaparak neredeyse her sorguda `confidence: 0.85` döndürdü. Bu, LLM'lerin in-context örnek değerlerini taklit etme eğiliminin bir örneğidir. Düzeltme: prompt örneğindeki sayısal değer `0.73`'e değiştirildi ve bir scoring guide eklendi.

### 1.2 OR Koşulunun Bypass Etkisi

`autorag_chain.py`'daki orijinal kabul koşulu:

```python
if grade["relevant"] or grade["confidence"] >= self._grade_threshold:
    break
```

Bu koşul, `relevant=True` olduğunda `grade_threshold`'u tamamen bypass ediyordu. qwen2.5:3b amnesty_qa chunk'larını her zaman `relevant=True` olarak değerlendirdi (beklenen bir davranış: chunk'lar gerçekten de konuyla ilgili). Sonuç: eşik değerinin hiçbir önemi kalmadı.

**Düzeltme:** Koşul şu şekilde değiştirildi:

```python
grade_ok = grade["confidence"] >= self._grade_threshold
score_ok = (self._retrieval_threshold <= 0.0) or (avg_score >= self._retrieval_threshold)
accepted = grade_ok and score_ok
```

`grade["relevant"]` artık kabul/reddetme kararında rol oynamıyor. Sadece `confidence` eşiği belirleyici.

---

## 2. Retrieval-Friendly Dataset Auto-RAG Avantajını Neden Gizledi?

amnesty_qa veri seti şu özelliklere sahip:
- Sorular ve belgeler aynı kaynaktan türetilmiş (Amnesty International raporları)
- Sorular belgelerdeki ifadelere yakın kelime örtüşmesi içeriyor
- BM25 exact-match büyük ölçüde başarılı

Bu koşullar altında:
- İlk retrieval zaten doğru chunk'ları getiriyor
- Grade node yüksek alaka skoru veriyor
- Rewrite mekanizmasının devreye gireceği bir senaryo oluşmuyor

Self-reflection, **başarısız retrieval durumları için tasarlanmış** bir güvenlik ağıdır. Retrieval başarısız olmadığında ağ gerrilmez.

---

## 3. Challenging Benchmark Neden Gerekli?

`data/challenging_qa_dataset.json` şu özelliklere sahip sorular içeriyor:

| Zorluk Türü | Örnek | Beklenen Etki |
|---|---|---|
| **Paraphrase** | "jurisprudential shift in reproductive rights law" | BM25 başarısız (keyword mismatch) |
| **Dolaylı** | "industrial entities bearing carbon burden" | Dense kısmi, BM25 başarısız |
| **Multi-hop** | Kombine iki konuyu ilişkilendiren | İlk retrieval kapsam dışı |
| **Teknik** | "humanitarian law frameworks for non-state actors" | Terminoloji farklılığı |

Bu sorularda BM25 keyword overlap'i düşük olacağı için `avg_retrieval_score` düşük gelecek; bu da `retrieval_threshold` parametresiyle rewrite tetiklenebilir.

---

## 4. Rewrite Karar Mekanizması Nasıl Çalışıyor?

```
Retrieve → Grade → [Kabul mı?] → Evet: Generate
                              → Hayır: Rewrite → Retrieve (tekrar)
```

Kabul koşulu (her attempt sonrası):
```
grade_confidence >= grade_threshold   (varsayılan: 0.75)
AND
avg_retrieval_score >= retrieval_threshold   (varsayılan: 0.0 = devre dışı)
```

Her iki koşul da sağlanırsa retrieval kabul edilir. Aksi halde:
1. `grade_result.reasoning` (neden başarısız) alınır
2. `rewrite_node` bu gerekçeyi kullanarak yeni sorgu üretir
3. Yeni sorguyla tekrar retrieve edilir
4. Maksimum `max_rewrites` kez tekrar (varsayılan: 2)

Her attempt detaylı `rewrite_trace`'e kaydedilir.

---

## 5. Threshold Değişimi Sistemi Nasıl Etkiliyor?

| `grade_threshold` | Beklenen Davranış |
|---|---|
| **0.60** | Çok lenient. amnesty_qa'da rewrite=0. Kolayda gereksiz. |
| **0.75** | Orta. Paraphrase sorgularda rewrite devreye girebilir. Önerilen. |
| **0.85** | Strict. Çoğu sorguda rewrite tetiklenir, latency artar. |

Threshold, **doğruluk-verimlilik dengesi**ni belirleyen ana parametredir. Düşük threshold → hızlı ama gözden kaçabilir. Yüksek threshold → yavaş ama daha az noise.

`retrieval_threshold` parametresi, LLM grade sonucundan bağımsız olarak düşük BM25/dense skorlarında rewrite zorlayabilir. Örnek:
- `retrieval_threshold=0.50` → BM25 ortalaması 0.50 altında kalırsa rewrite, grade ne derse desin

---

## 6. Deney Çalıştırma

```bash
# Sadece threshold deneyi (challenging dataset ile)
python -m eval.run_full_benchmark \
    --only thresholds \
    --dataset data/challenging_qa_dataset.json \
    --n-ablation 10

# Ablation study
python -m eval.run_full_benchmark \
    --only ablation \
    --dataset data/challenging_qa_dataset.json \
    --n-ablation 10

# Rewrite effectiveness
python -m eval.run_full_benchmark \
    --only rewrite \
    --dataset data/challenging_qa_dataset.json \
    --n-ablation 10
```

Sonuçlar `results/` altında:
- `benchmark_thresholds.json` — threshold=0.60/0.75/0.85 karşılaştırması
- `ablation_study.json` — 3 koşul ablation
- `rewrite_effectiveness.json` — per-question rewrite öncesi/sonrası
