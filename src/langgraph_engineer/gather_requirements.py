from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState
from typing import TypedDict
from langchain_core.messages import RemoveMessage

gather_prompt = """You are tasked with helping build LangGraph applications. \
LangGraph is a framework for developing LLM applications. \
It represents agents as graphs. These graphs can contain cycles and often contain branching logic.

Your first job is to gather all the user requirements about the topology of the graph. \
You should have a clear sense of all the nodes of the graph/agent, and all the edges. 

You are conversing with a user. Ask as many follow up questions as necessary - but only ask ONE question at a time. \
Only gather information about the topology of the graph, not about the components (prompts, LLMs, vector DBs). \
If you have a good idea of what they are trying to build, call the `Build` tool with a detailed description.

Do not ask unnecessary questions! Do not ask them to confirm your understanding or the structure! The user will be able to \
correct you even after you call the Build tool, so just do enough to get an MVP."""


class Build(TypedDict):
    requirements: str


def gather_requirements(state: AgentState, config):
    messages = [
       {"role": "system", "content": gather_prompt}
   ] + state['messages']
    model = _get_model(config, "openai-mini", "gather_model").bind_tools([Build])
    response = model.invoke(messages)
    if len(response.tool_calls) == 0:
        return {"messages": [response]}
    else:
        requirements = response.tool_calls[0]['args']['requirements']
        delete_messages = [RemoveMessage(id=m.id) for m in state['messages']]
        return {"requirements": requirements, "messages": delete_messages}
