import streamlit as st
import os

# Ensure your environment variables are set if needed
# os.environ["RAPIDAPI_KEY"] = "YOUR_RAPIDAPI_KEY"

from agent import graph
from state import AgentState, GraphConfig
from langchain_core.messages import UserMessage, AIMessage, SystemMessage

# Set Streamlit page configuration
st.set_page_config(page_title="LangGraph Real Estate Assistant", page_icon="🏠")

# Initialize session state
if "messages" not in st.session_state:
    # We'll store a list of messages in the format accepted by our graph.
    # Each message is a dict: {"role": "user"/"system"/"assistant", "content": str}
    # Compatible with langchain_core.messages.
    st.session_state.messages = []

if "requirements" not in st.session_state:
    st.session_state.requirements = None

if "config" not in st.session_state:
    # Set default configuration. Adjust model choices as per your available configs.
    st.session_state.config = GraphConfig(
        gather_model="openai-mini",
        api_call_builder="openai-mini",
        writer="openai-mini",
        report_writer="openai-mini",
        report_saver="openai-mini",
    )

def run_graph(messages, requirements=None):
    """
    Runs the LangGraph workflow with the given messages and requirements.
    Returns the updated state after processing.
    """
    # Construct the AgentState
    # AgentState inherits from MessagesState and may also hold requirements
    # According to your code, AgentState may have fields: requirements, code, accepted, report
    # We'll provide what's known and defaults for the rest.
    # The graph expects 'messages' in the state as well.
    state = AgentState(
        messages=messages,
        requirements=requirements if requirements else None,
        code=None,
        accepted=False,
        report=None
    )

    # Run the graph
    result = graph.run(state, st.session_state.config)

    # The result is an OutputState or an updated AgentState after processing.
    # According to your code steps, some steps return updated keys like messages, requirements, report, etc.
    # result may contain keys like "messages", "requirements", "report".
    # We'll extract these and update session state.
    if "requirements" in result:
        st.session_state.requirements = result["requirements"]
    if "report" in result:
        # The final report might be generated at the end. Store it if needed.
        # For now, we'll just keep it in session state.
        st.session_state.report = result["report"]
    if "messages" in result:
        # The result might contain a new model response. Add it to session messages.
        # This typically would be something like a single AIMessage or a list of messages.
        responses = result["messages"]
        # responses could be a list of Message-like objects (e.g., AIMessage) returned by the invocation
        # We need to convert them back to a standard dict format if necessary.
        for r in responses:
            # If r is an object with role & content, adjust as needed
            # Check if `r` is already a dict or a message object
            if isinstance(r, AIMessage):
                st.session_state.messages.append({"role": "assistant", "content": r.content})
            elif isinstance(r, UserMessage):
                st.session_state.messages.append({"role": "user", "content": r.content})
            elif isinstance(r, SystemMessage):
                st.session_state.messages.append({"role": "system", "content": r.content})
            elif isinstance(r, dict) and "role" in r and "content" in r:
                # If it's already a dict
                st.session_state.messages.append(r)
    # Return nothing explicitly; session state is updated globally.


st.title("🏠 Real Estate Report Assistant")

# Display conversation history
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.markdown(f"**Assistant**: {msg['content']}")
    elif msg["role"] == "user":
        st.markdown(f"**You**: {msg['content']}")
    elif msg["role"] == "system":
        # System messages can be hidden or shown differently
        st.markdown(f"_System_: {msg['content']}")

# User input box
user_input = st.text_input("Your message:", value="", placeholder="Ask a question or provide details...")
if st.button("Send", type="primary"):
    if user_input.strip():
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Run the graph with updated messages and requirements
        run_graph(st.session_state.messages, st.session_state.requirements)

        # Clear input after sending
        st.experimental_rerun()
