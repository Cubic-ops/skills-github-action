import os
import getpass
from langchain.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
class State(TypedDict):
    topic: str
    joke: str
    poem:str
    story:str
    combined_output:str

def call_llm_1(state:State):
    """
    Docstring for call_llm_1
    
    :param state: Description
    :type state: State
    """
    msg=llm.invoke(f"Write a short joke about{state['topic']}")
    return {"joke":msg.content}
def call_llm_2(state:State):
    """
    Docstring for call_llm_2
    
    :param state: Description
    :type state: State
    """
    msg=llm.invoke(f"Write a short poem about{state['topic']}")
    return {"poem":msg.content}
def call_llm_3(state:State):
    """
    Docstring for call_llm_3
    
    :param state: Description
    :type state: State
    """
    msg=llm.invoke(f"Write a short story about{state['topic']}")
    return {"story":msg.content}
def aggregate_outputs(state:State):
    """
    combine the outputs from poen, story and joke into a single string
    """
    combined_output=f"Joke:\n{state['joke']}\n\nPoem:\n{state['poem']}\n\nStory:\n{state['story']}"
    return {"combined_output":combined_output}

parellel_builder=StateGraph(State)

parellel_builder.add_node("call_llm_1",call_llm_1)
parellel_builder.add_node("call_llm_2",call_llm_2)
parellel_builder.add_node("call_llm_3",call_llm_3)
parellel_builder.add_node("aggregate_outputs",aggregate_outputs)
parellel_builder.add_edge(START,"call_llm_1")
parellel_builder.add_edge(START,"call_llm_2")
parellel_builder.add_edge(START,"call_llm_3")
parellel_builder.add_edge("call_llm_1","aggregate_outputs")
parellel_builder.add_edge("call_llm_2","aggregate_outputs")
parellel_builder.add_edge("call_llm_3","aggregate_outputs")
parellel_builder.add_edge("aggregate_outputs",END)
parallel_agent=parellel_builder.compile()

if __name__=="__main__":
    result=parallel_agent.invoke({"topic":"artificial intelligence"})
    print(result["combined_output"])