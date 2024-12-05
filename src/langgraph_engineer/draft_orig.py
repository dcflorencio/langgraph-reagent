from langgraph_engineer.loader import load_github_file
from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState

prompt = """You are tasked with answering questions about LangGraph functionality and bugs.
Here is a long unit test file for LangGraph. This should contain a lot (but possibly not all) \
relevant information on how to use LangGraph.

<unit_test_file>
{file}
</unit_test_file>

Based on the information above, attempt to answer the user's questions. If you generate a code block, only \
generate a single code block - eg lump all the code together (rather than splitting up). \
You should encode helpful comments as part of that code block to understand what is going on. \
ALWAYS just generate the simplest possible example - don't make assumptions that make it more complicated. \
For "messages", these are a special object that looks like: {{"role": .., "content": ....}}

If users ask for a messages key, use MessagesState which comes with a built in `messages` key. \
You can import MessagesState from `langgraph.graph` and it is a TypedDict, so you can subclass it and add new keys to use as the graph state.

Make sure any generated graphs have at least one edge that leads to the END node - you need to define a stopping criteria!

You generate code using markdown python syntax, eg:

```python
...
```

Remember, only generate one of those code blocks!"""


def draft_answer(state: AgentState, config):
    github_url = "https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/tests/test_pregel.py"
    file_contents = load_github_file(github_url)
    messages = [
        {"role": "system", "content": prompt.format(file=file_contents)},
                   {"role": "user", "content": state.get('requirements')}
    ] + state['messages']
    model = _get_model(config, "openai-mini", "draft_model")
    response = model.invoke(messages)
    return {"messages": [response]}
