from flask import Flask
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

@app.route('/')
def hello():

    # Create the agent
    memory = MemorySaver()
    model = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")
    search = TavilySearchResults(max_results=2)
    tools = [search]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

    # Use the agent
    config = {"configurable": {"thread_id": "abc123"}}
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="Salut, j'habite à saint mars du désert")]}, config
    ):
        print(chunk)
        print("----")

    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="quel age à Messi ? Le joueur de foot")]}, config
    ):
        print(chunk)
        print("----")
        
    return 'Hello Langchain!'