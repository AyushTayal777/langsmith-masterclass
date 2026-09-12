
import os
import requests

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

os.environ['LANGCHAIN_PROJECT']='Agent'

load_dotenv()

# DuckDuckGo search tool
search_tool = DuckDuckGoSearchRun()


# Weather tool
@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city.
    """

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    url = (
        f"https://api.weatherstack.com/current"
        f"?access_key={api_key}"
        f"&query={city}"
    )

    response = requests.get(url)

    return response.text


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)


agent = create_agent(
    model=llm,
    tools=[
        search_tool,
        get_weather_data
    ]
)


response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Identify the birthplace city of Kalpana Chawla (search) and give its current temperature."
            }
        ]
    }
)

print(response)

print("\nFinal Answer:")
print(response["messages"][-1].content)

