"""
Standart deney sonuç şeması.

Tüm eval modülleri (eval_runner, retrieval_experiment, chunking_experiment)
bu şemayı kullanarak sonuçlarını JSON'a yazar. Streamlit dashboard bu formatı okur.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from typing import Optional


@dataclass
class ExperimentResult:
    experiment_name: str
    experiment_type: str  # "comparison" | "retrieval" | "chunking"

    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    # Model bilgisi
    model_name: Optional[str] = None
    provider: Optional[str] = None

    # Retrieval ayarları
    retriever_type: Optional[str] = None
    chunk_size: Optional[int] = None
    top_k: Optional[int] = None

    # RAGAS metrikleri
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None

    # Gecikme
    latency_seconds: Optional[float] = None
    avg_latency_seconds: Optional[float] = None

    # Self-reflection istatistikleri
    rewrite_count: int = 0
    avg_rewrite_count: float = 0.0

    # Maliyet
    token_usage: Optional[int] = None
    cost_estimate: Optional[float] = None

    # Veri seti boyutu
    n_questions: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentResult":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
