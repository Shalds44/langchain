from langgraph.prebuilt import tools_condition
from nodes import grade_documents

def setup_edges(workflow):
    # Si la tools_condition retourne tools on va vers retrieve si le worflow se termine.
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "retrieve", END: END},
    )
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate", END)
    # après rewrite le workflow revient à agent
    workflow.add_edge("rewrite", "agent")
