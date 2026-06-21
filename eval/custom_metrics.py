"""
Custom RAGAS-style metric computation using direct Ollama API calls.

Avoids the ragas library's dependency on transformers/torch for embeddings,
and uses qwen2.5:3b for fast faithfulness/precision/recall judgments.
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

import ollama

logger = logging.getLogger(__name__)

_EMBED_MODEL = "qwen2.5:3b"
_JUDGE_MODEL = "qwen2.5:3b"
_GEN_MODEL = "qwen2.5:3b"
_OLLAMA_HOST = "http://localhost:11434"


def _embed(texts: list[str]) -> list[list[float]]:
    client = ollama.Client(host=_OLLAMA_HOST)
    resp = client.embed(model=_EMBED_MODEL, input=texts)
    return [list(e) for e in resp.embeddings]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-10)


def _judge(prompt: str) -> str:
    client = ollama.Client(host=_OLLAMA_HOST)
    resp = client.chat(
        model=_JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict": 50},
    )
    return resp.message.content.strip()


def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Faithfulness: fraction of answer sentences that are grounded in retrieved contexts.
    A sentence is grounded if the context contains the same or equivalent information.
    """
    ctx_text = "\n\n".join(contexts[:5])
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 25]
    if not sentences:
        return 1.0

    supported = 0
    for sent in sentences:
        prompt = (
            "Does the context below contain information that supports or is consistent with "
            "the following statement? The exact wording does not need to match — semantic "
            "equivalence is sufficient.\n\n"
            f"Statement: {sent}\n\n"
            f"Context:\n{ctx_text}\n\n"
            "Answer with only 'yes' or 'no'."
        )
        try:
            reply = _judge(prompt).lower()
            if reply.startswith("yes"):
                supported += 1
        except Exception:
            supported += 1
    return round(supported / len(sentences), 4)


def compute_answer_relevancy(question: str, answer: str, n: int = 3) -> float:
    """
    Answer Relevancy: generates n questions from the answer, measures cosine
    similarity between generated questions and the original question.
    """
    client = ollama.Client(host=_OLLAMA_HOST)
    prompt = (
        f"Generate {n} questions that this answer is trying to answer. "
        f"Output one question per line, no numbering.\n\nAnswer: {answer}"
    )
    try:
        resp = client.chat(
            model=_GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 200},
        )
        raw = resp.message.content
        # Strip <think>...</think> from deepseek-r1
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        gen_questions = [l.strip() for l in raw.splitlines() if l.strip()][:n]
    except Exception:
        return 0.0

    if not gen_questions:
        return 0.0

    all_texts = [question] + gen_questions
    try:
        embeddings = _embed(all_texts)
    except Exception:
        return 0.0

    q_emb = embeddings[0]
    sims = [_cosine(q_emb, gen_emb) for gen_emb in embeddings[1:]]
    return round(sum(sims) / len(sims), 4)


def compute_context_precision(question: str, contexts: list[str], ground_truth: str) -> float:
    """
    Context Precision: fraction of retrieved contexts that are relevant to answering the question.
    """
    if not contexts:
        return 0.0

    relevant = 0
    for ctx in contexts:
        prompt = (
            "Is the following context useful for answering the question?\n"
            f"Question: {question}\n"
            f"Context: {ctx[:500]}\n\n"
            "Reply with only 'yes' or 'no'."
        )
        try:
            reply = _judge(prompt).lower()
            if reply.startswith("yes"):
                relevant += 1
        except Exception:
            relevant += 1
    return round(relevant / len(contexts), 4)


def compute_context_recall(contexts: list[str], ground_truth: str) -> float:
    """
    Context Recall: fraction of ground truth claims covered by retrieved contexts.
    """
    ctx_text = "\n".join(contexts[:5])
    sentences = [s.strip() for s in re.split(r"[.!?]", ground_truth) if len(s.strip()) > 15]
    if not sentences:
        return 1.0

    covered = 0
    for sent in sentences:
        prompt = (
            "Is the following information present in the context below?\n"
            f"Information: {sent}\n"
            f"Context:\n{ctx_text[:1500]}\n\n"
            "Reply with only 'yes' or 'no'."
        )
        try:
            reply = _judge(prompt).lower()
            if reply.startswith("yes"):
                covered += 1
        except Exception:
            covered += 1
    return round(covered / len(sentences), 4)


def compute_all_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """
    Compute all four RAGAS-style metrics for a list of chain output records.

    Each record must have: question, answer, contexts (list[str]), ground_truth.
    """
    if not records:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}

    faith_scores: list[float] = []
    rel_scores: list[float] = []
    prec_scores: list[float] = []
    recall_scores: list[float] = []

    for i, rec in enumerate(records):
        q = rec["question"]
        a = rec.get("answer", "")
        ctxs = rec.get("contexts", [])
        gt = rec.get("ground_truth", "")

        logger.info("  [%d/%d] Evaluating: %s", i + 1, len(records), q[:60])

        if a and ctxs:
            faith_scores.append(compute_faithfulness(a, ctxs))
            rel_scores.append(compute_answer_relevancy(q, a))
            prec_scores.append(compute_context_precision(q, ctxs, gt))
        if ctxs and gt:
            recall_scores.append(compute_context_recall(ctxs, gt))

    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "faithfulness": _avg(faith_scores),
        "answer_relevancy": _avg(rel_scores),
        "context_precision": _avg(prec_scores),
        "context_recall": _avg(recall_scores),
    }
