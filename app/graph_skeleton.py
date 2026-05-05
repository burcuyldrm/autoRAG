import numpy as np

from app.schemas import GraphState
from app.retriever import VectorRetriever
from app.hybrid_retriever import HybridRetriever
from app.config import GRADE_THRESHOLD, MAX_ITERATIONS


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
    # Şimdilik placeholder.
    # T14 merge edilince burada gerçek grading sonucu kullanılacak.
    state["confidence"] = state.get("confidence", 1.0)
    return state


def route_node(state: GraphState) -> str:
    confidence = state.get("confidence", 0.0)
    iteration = state.get("iteration", 0)

    if confidence < GRADE_THRESHOLD and iteration < MAX_ITERATIONS:
        return "rewrite"

    return "generate"


def rewrite_node(state: GraphState) -> GraphState:
    state["rewritten_query"] = state.get("query", "")
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

        route = route_node(state)

        if route == "rewrite":
            state["iteration"] = state.get("iteration", 0) + 1
            state = self.nodes["rewrite"](state)
            state = self.nodes["generate"](state)
        else:
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
        "mode": "vector",
        "confidence": 1.0,
        "iteration": 0,
    }
def rewrite_node(state: GraphState) -> GraphState:
    query = state.get("query", "")

    # basit iyileştirme (şimdilik)
    improved_query = query + " detailed explanation"

    state["rewritten_query"] = improved_query
    state["iteration"] = state.get("iteration", 0) + 1

    return state