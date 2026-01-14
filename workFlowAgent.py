import os
import getpass
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage

from langchain_anthropic import ChatAnthropic

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
# Schema for structured output
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    query:str=Field(None,description="Query that is optimized web search.")
    justification:str=Field(None,description="Why is this query related with this user's original question?")
structured_llm=llm.with_structured_output(SearchQuery)
output=structured_llm.invoke([
                    SystemMessage(content="Route the input to story, joke, or poem based on the user's request."),
                    HumanMessage(content="How does Calcium CT score relate to high cholesterol?")
                    ])
print(output)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b

llm_with_tools = llm.bind_tools([multiply])
tools=[multiply]
# Invoke the LLM with input that triggers the tool call
msg = llm_with_tools.invoke("What is 2 times 3?")
# Get the tool call
print(msg.content)
# tool_call = msg.tool_calls[0]
# # Find the tool implementation by name
# result=tools[0].invoke(tool_call["args"])
# print(result)
