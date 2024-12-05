from typing import Literal

from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import AIMessage

from langgraph.prebuilt import ToolNode


from langgraph_engineer.check import check
from langgraph_engineer.critique import critique
from langgraph_engineer.draft import draft_answer, fetch_zillow_data
from langgraph_engineer.gather_requirements import gather_requirements
from langgraph_engineer.state import AgentState, OutputState, GraphConfig



def route_critique(state: AgentState) -> Literal["draft_answer", END]:
    if state['accepted']:
        return END
    else:
        return "draft_answer"

def route_check(state: AgentState) -> Literal["critique", "draft_answer"]:
    if isinstance(state['messages'][-1], AIMessage):
        return "critique"
    else:
        return "draft_answer"


def route_start(state: AgentState) -> Literal["draft_answer", "gather_requirements"]:
    if state.get('requirements'):
        return "draft_answer"
    else:
        return "gather_requirements"


def route_gather(state: AgentState) -> Literal["draft_answer", END]:
    if state.get('requirements'):
        return "draft_answer"
    else:
        return END


# Define a new graph
workflow = StateGraph(AgentState, input=MessagesState, output=OutputState, config_schema=GraphConfig)
workflow.add_node(gather_requirements)
workflow.add_node(draft_answer, ToolNode([fetch_zillow_data]))

workflow.add_node("writer", ToolNode([fetch_zillow_data]))
# workflow.add_node(critique)
# workflow.add_node(check)
workflow.set_conditional_entry_point(route_start)
workflow.add_conditional_edges("gather_requirements", route_gather)
# workflow.add_edge("draft_answer", "check")
# workflow.add_conditional_edges("check", route_check)
# workflow.add_conditional_edges("critique", route_critique)
workflow.add_edge("draft_answer", "writer")
workflow.add_edge("writer", END)
graph = workflow.compile()
