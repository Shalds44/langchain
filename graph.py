from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from agent_state import AgentState
from nodes import agent, rewrite, generate
from edges import setup_edges
from retriever import tools

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
retrieve = ToolNode([tools])
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)
workflow.add_edge(START, "agent")
setup_edges(workflow)
graph = workflow.compile()
