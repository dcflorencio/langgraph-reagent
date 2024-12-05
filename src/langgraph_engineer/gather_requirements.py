from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState
from typing import TypedDict
from langchain_core.messages import RemoveMessage

gather_prompt = """
You are a professional assistant tasked with gathering requirements to generate customized real estate reports. 
Your job is to ask the user relevant questions to collect all necessary details, including:

- Whether they are looking to buy or rent.
- If they are interested in a house, apartment, or other type of property.
- The number of bedrooms and bathrooms they need.
- Their preferred locations, city or neighborhoods
- The desired square footage or size of the property.
- Their budget or price range.

Ask one question at a time, and only ask for clarification if the user’s input is unclear or incomplete. 
Avoid asking unnecessary questions or gathering information unrelated to the real estate report. 
Once you have enough details, summarize the collected requirements and confirm readiness to proceed with generating the report.

Do not ask unnecessary questions! Do not ask them to confirm your understanding or the structure! The user will be able to 
correct you after.
"""


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
