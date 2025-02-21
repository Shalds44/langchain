from flask import Flask
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from agent_state import AgentState
from dotenv import load_dotenv
import qdrant_client
import os

load_dotenv()

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello Langchain!'

@app.route('/claudeSimple')
def claude():

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
        {"messages": [HumanMessage(content="l'âge de Messi ? ")]}, config
    ):
        print(chunk)
        print("----")
        
    return 'Hello Claude agent!'

@app.route('/rag')
def rag():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # url = "https://c5a5059f-5a66-4dc4-a644-54f96c845264.europe-west3-0.gcp.cloud.qdrant.io"
    # api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ3MTUzNTE4fQ.wxvYsEuU_llKYGNDWxJL5V7UwGNlYGmZr6_GGJ1nDb4"
    # collection_name = "my_documents"
    
    qdrant = qdrant_client.QdrantClient(url=url, api_key=api_key, prefer_grpc=True)
    
    # Vérification de l'existence de la collection
    collections = qdrant.get_collections().collections
    collection_exists = any(collection.name == collection_name for collection in collections)
    
    if collection_exists:
        # Charger la collection existante
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name,
            url=url,
            prefer_grpc=True,
            api_key=api_key,
        )
        print("Collection chargée à partir de Qdrant.")
        
    else:
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
        ]
        
        # Charger les documents
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Diviser les documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)
    
        qdrant = QdrantVectorStore.from_documents(
            doc_splits,
            embeddings,
            url=url,
            prefer_grpc=True,
            api_key=api_key,
            collection_name="my_documents",
        )
        print("Nouvelle collection créée dans Qdrant.")
    
    retriever = qdrant.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
    )

    tools = [retriever_tool]
        
    return 'Hello rag!'