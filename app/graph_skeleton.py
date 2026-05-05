


# -------- NODELAR --------

def retrieve_node(state: GraphState) -> GraphState:


    return state


def grade_node(state: GraphState) -> GraphState:

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