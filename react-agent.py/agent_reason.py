from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool,create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langchain import hub
model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

search_tool=TavilySearchResults(search_dept="basic")

@tool
def get_system_time(format:str="%y-%m-%d %H:%M:%S"):
    """ Returns the current date and time  in a specified format"""

    current_time=datetime.datetime.now()

    formatted_time=current_time.strftime(format=format)

    return formatted_time

tools=[search_tool,get_system_time]

react_prompt=hub.pull("hwchase17/react")

react_agent_runnable=create_react_agent(tools=tools,prompt=react_prompt,llm=model)