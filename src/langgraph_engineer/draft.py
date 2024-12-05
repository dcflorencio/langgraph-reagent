from langgraph_engineer.model import _get_model
from langgraph_engineer.state import AgentState

from typing import TypedDict
from langchain_core.messages import RemoveMessage
import os
from typing import Any, Dict
import http.client

rapidapi_key = os.getenv("RAPIDAPI_KEY")

prompt = """
you are tasked to make an api call using fetch_zillow_data to get infromation on real estate according to the requirements gathered

Function to query Zillow API.
All parameters are optional.

    :param location: Location to search (e.g., 'San Francisco').
    :param page: Page number of the result set (default is 1).
    :param status_type: Property status ('ForSale', 'ForRent').
    :param home_type: Type of home (e.g., if status_type = ForSale then 'Apartments', 'Houses', 'Condos', 'Townhomes' ; if status_type = ForRent then 'Houses', 'Townhomes', 'Apartments_Condos_Co-ops'; ).
    :param sort: Sorting option (e.g., if status_type = ForSale then 'Price_Low_High'; if status_type = ForRent then 'Payment_Low_High').
    :param min_price: Minimum price.
    :param max_price: Maximum price.
    :param rent_min_price: Minimum rent price.
    :param rent_max_price: Maximum rent price.
    :param baths_min: Minimum number of bathrooms.
    :param baths_max: Maximum number of bathrooms.
    :param beds_min: Minimum number of bedrooms.
    :param beds_max: Maximum number of bedrooms.
    :param sqft_min: Minimum square footage.
    :param sqft_max: Maximum square footage.
    :param days_on: Number of days on the market.
    :param sold_in_last: Filter for sold in the last 'x' days.
    :param keywords: Keywords for search (e.g., 'modern house 2 stories').
    :return: A dictionary containing the Zillow API response.

"""


def fetch_zillow_data(location: str = None, page: int = 1, status_type: str = None, home_type: str = None, sort: str = None,
                      min_price: int = None, max_price: int = None, rent_min_price: int = None, rent_max_price: int = None,
                      baths_min: int = None, baths_max: int = None, beds_min: int = None, beds_max: int = None,
                      sqft_min: int = None, sqft_max: int = None, days_on: int = None, sold_in_last: int = None,
                      keywords: str = None) -> Dict[str, Any]:
    """
    Function to query Zillow API.
    All parameters are optional.


    
    :return: A dictionary containing the Zillow API response.
    """
    
    # connection to Zillow API via HTTPS
    conn = http.client.HTTPSConnection("zillow-com1.p.rapidapi.com")

    # API key and host information
    headers = {
        'x-rapidapi-key': rapidapi_key,
        'x-rapidapi-host': "zillow-com1.p.rapidapi.com"
    }

    # Prepare query parameters, only include if not None
    query_params = []
    if location:
        query_params.append(f"location={location.replace(' ', '%20')}")
    query_params.append(f"page={page}")
    if status_type:
        query_params.append(f"status_type={status_type}")
    if home_type:
        query_params.append(f"home_type={home_type}")
    if sort:
        query_params.append(f"sort={sort}")
    if min_price is not None:
        query_params.append(f"minPrice={min_price}")
    if max_price is not None:
        query_params.append(f"maxPrice={max_price}")
    if rent_min_price is not None:
        query_params.append(f"rentMinPrice={rent_min_price}")
    if rent_max_price is not None:
        query_params.append(f"rentMaxPrice={rent_max_price}")
    if baths_min is not None:
        query_params.append(f"bathsMin={baths_min}")
    if baths_max is not None:
        query_params.append(f"bathsMax={baths_max}")
    if beds_min is not None:
        query_params.append(f"bedsMin={beds_min}")
    if beds_max is not None:
        query_params.append(f"bedsMax={beds_max}")
    if sqft_min is not None:
        query_params.append(f"sqftMin={sqft_min}")
    if sqft_max is not None:
        query_params.append(f"sqftMax={sqft_max}")
    if days_on is not None:
        query_params.append(f"daysOn={days_on}")
    if sold_in_last is not None:
        query_params.append(f"soldInLast={sold_in_last}")
    if keywords:
        query_params.append(f"keywords={keywords.replace(' ', '%20')}")

    # final query string
    query_string = "?" + "&".join(query_params)

    # GET request with headers and query parameters
    conn.request("GET", f"/propertyExtendedSearch{query_string}", headers=headers)

    # Get the response from the API
    response = conn.getresponse()
    data = response.read()

    print(data.decode("utf-8"))

    # Decode and return the response data as a dictionary
    return {
        "status": response.status,
        "reason": response.reason,
        "data": data.decode("utf-8")
    }

class Build(TypedDict):
    requirements: str
    data: str


def draft_answer(state: AgentState, config):
    messages = [
       {"role": "system", "content": prompt}, 
       {"role": "user", "content": state.get('requirements')}
   ] + state['messages']
    model = _get_model(config, "openai-mini", "gather_model").bind_tools([fetch_zillow_data])
    response = model.invoke(messages)
    if len(response.tool_calls) == 0:
        return {"messages": [response]}
    else:
        data = response.tool_calls
        delete_messages = [RemoveMessage(id=m.id) for m in state['messages']]
        return {"messages": [response]}


        # return {"data": data, "messages": delete_messages}