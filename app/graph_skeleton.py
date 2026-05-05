from typing import Dict, Any
import numpy as np

from app.schemas import GraphState
from app.retriever import VectorRetriever
from app.hybrid_retriever import HybridRetriever


# -------- NODELAR --------

def retrieve_node(state: GraphState) -> GraphState:
    query_embedding = np.array([1.0, 0.0])

    chunks = [
        {"text": "apple"},
        {"text": "banana"},
        {"text": "car"},
    ]

    embeddings = [
        np.array([1.0, 0.0]),
        np.array([0.9, 0.1]),
        np.array([0.0, 1.0]),
    ]

    retriever = VectorRetriever(chunks, embeddings)
    vector_results = retriever.retrieve(query_embedding, k=3)

    mode = state.get("mode", "vector")

    if mode == "hybrid":
        bm25_results = [
            {"text": "banana"},
            {"text": "apple"},
        ]

        hybrid = HybridRetriever(vector_results, bm25_results)
        results = hybrid.fuse(top_k=3)
    else:
        results = vector_results

    state["retrieved_chunks"] = results

    return state


def grade_node(state: GraphState) -> GraphState:
    return state


def rewrite_node(state: GraphState) -> GraphState:
    return state


def generate_node(state: GraphState) -> GraphState:
    state["answer"] = "placeholder answer"
    return state


# -------- GRAPH --------

class StateGraph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, name: str, func):
        self.nodes[name] = func

    def run(self, state: GraphState) -> GraphState:
        state = self.nodes["retrieve"](state)
        state = self.nodes["grade"](state)
        state = self.nodes["rewrite"](state)
        state = self.nodes["generate"](state)
        return state


def build_graph() -> StateGraph:
    graph = StateGraph()

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)

    return graph


# -------- INITIAL STATE --------

def create_initial_state(query: str) -> GraphState:
    return {
        "query": query,
        "rewritten_query": "",
        "retrieved_chunks": [],
        "graded_chunks": [],
        "answer": "",
        "mode": "vector",  # default
    }