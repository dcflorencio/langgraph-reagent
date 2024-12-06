from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState
from typing import TypedDict
from langchain_core.messages import RemoveMessage

writer_prompt = """
Write a detailed report on the properties. make it in markdown format
"""


class Build_write(TypedDict):
    report: str


def report_writer(state: AgentState, config):
    messages = [
       {"role": "system", "content": writer_prompt}
   ] + state['messages']
    model = _get_model(config, "openai-mini", "gather_model").bind_tools([Build_write])
    response = model.invoke(messages)
    if len(response.tool_calls) == 0:
        return {"no tool call"}
    else:
        return {"messages": [response]}
