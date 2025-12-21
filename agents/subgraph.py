from typing import TypedDict,Annotated
from langgraph.graph import add_messages,StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode


load_dotenv()



class ChildState:
    messages:Annotated[list,add_messages]


search_tool=TavilySearchResults(max_results=2)

tool=[search_tool]

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

llm_with_tools=model.bind_tools(tools=tool)

graph=StateGraph()

tool_node=ToolNode(tools=tool)
def agent(state:ChildState):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

def tool_router(state:ChildState):
    last_message=state['messages'][-1]
    if(hasattr(last_message,"tool_calls")):
      return "tool_node"
    else:
        return END
    
graph.add_node("agent",agent)
graph.add_node("tool_node",tool_node)
graph.set_entry_point("agent")

graph.add_conditional_edges("agent",tool_router)
graph.add_edge("tool_node","agent")

search_app=graph.compile()

search_app

from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod

display(
    Image(
        search_app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API
        )
    )
)