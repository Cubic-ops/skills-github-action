from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
import os
from langgraph.types import Command
import getpass
from langchain_anthropic import ChatAnthropic

os.environ["TRANSFORMERS_CACHE"] = os.path.join('E:\Transformer_cache', ".cache")
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)


model = init_chat_model("claude-haiku-4-5-20251001", temperature=0)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b 


@tool
def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b


tools = [add, multiply, divide]
tools_by_name = {tool_impl.name: tool_impl for tool_impl in tools}
model_with_tools = model.bind_tools(tools)


def llm_call(state: MessagesState) -> dict:
    """Ask the model whether to answer or call a tool."""
    print(model_with_tools.invoke(
                [SystemMessage(content="You are a helpful arithmetic assistant.")
                 ]
                + state["messages"]
            ))
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content="You are a helpful arithmetic assistant.")
                 ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: MessagesState) -> dict:
    """Execute any tool calls produced by the model."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {"messages": []}

    result: list[ToolMessage] = []
    for tool_call in last_message.tool_calls:
        tool_impl = tools_by_name[tool_call["name"]]
        observation = tool_impl.invoke(tool_call["args"])
        result.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Route to the tool node if the model asked for tools."""
    # print(f"Deciding whether to continue with state: {len(state["messages"])}")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END],
)
agent_builder.add_edge("tool_node", "llm_call")
agent = agent_builder.compile()


if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [HumanMessage(content="Add 3 and 4.")]}
    )
    for message in result["messages"]:
        print(f"[{message.type}] {message.content}")
