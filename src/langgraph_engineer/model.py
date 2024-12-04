from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


def _get_model(config, default, key):
    model = config['configurable'].get(key, default)
    if model == "openai":
        return ChatOpenAI(temperature=0, model_name="gpt-4o")
    
    elif model == "openai-mini":
        return ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    elif model == "anthropic":
        return ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20240620")
    
    else:
        raise ValueError
