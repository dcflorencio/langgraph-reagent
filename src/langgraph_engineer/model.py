from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
load_dotenv()

def _get_model(config, default, key):
    model = config['configurable'].get(key, default)
    openai_api_key = os.getenv('OPENAI_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if model == "openai":
        return ChatOpenAI(temperature=0, model_name="gpt-4o",api_key= openai_api_key)
    
    elif model == "openai-mini":
        return ChatOpenAI(temperature=0, model_name="gpt-4o-mini",api_key=openai_api_key)

    elif model == "anthropic":
        return ChatAnthropic(temperature=0, model_name="claude-3-5-sonnet-20240620",api_key=anthropic_api_key)
    
    else:
        raise ValueError
