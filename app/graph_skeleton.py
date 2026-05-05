from typing import Dict, Any
from app.schemas import GraphState


# -------- NODELAR --------

def retrieve_node(state: GraphState) -> GraphState:
    # TODO: gerçek retriever eklenecek
    return state


def grade_node(state: GraphState) -> GraphState:
    # TODO: LLM ile relevance/faithfulness kontrolü
    return state


def rewrite_node(state: GraphState) -> GraphState:
    # TODO: query rewrite
    return state


def generate_node(state: GraphState) -> GraphState:
    # TODO: final answer üretimi
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
    }