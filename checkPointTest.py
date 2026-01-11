from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add
from IPython.display import Image, display

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
print(graph.invoke({"foo": "", "bar":[]}, config))
png_bytes = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)
print("Saved graph image to graph.png")
config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "1f0f34e2-f034-66b2-8002-417f6837cbda",
        "checkpoint_ns": ""
    }
}
# print(graph.get_state(config))
# get a state snapshot 
config = {"configurable": {"thread_id": "1"}}
print(list(graph.get_state_history(config)))