from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState
from typing import TypedDict
from langchain_core.messages import RemoveMessage
import os

# saver_prompt = """
# Save the markdown report locally. display the path you save it to.
# """


# class Build_write(TypedDict):
#     report: str


# def report_saver(state: AgentState, config):
#     messages = [
#        {"role": "system", "content": saver_prompt}
#    ] + state['report']
#     model = _get_model(config, "openai-mini", "report_saver").bind_tools([Build_write])
#     response = model.invoke(messages)

#     return {"messages": [response]}

def report_saver(state: AgentState, config):
    messages =  state['report'][0]

    file_name = "RE_report.md"
    
    # Save the file in the current path
    file_path = os.path.join(os.getcwd(), file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(messages)
    