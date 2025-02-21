from typing import Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition

def grade_documents(state) -> Literal["generate", "rewrite"]:
    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    llm_with_tool = model.with_structured_output(Grade)
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question.

        Here is the retrieved document:

        {context}

        Here is the user question:

        {question}

        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.

        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    chain = prompt | llm_with_tool
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score
    return "generate" if score == "yes" else "rewrite"

def agent(state):
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}

def rewrite(state):
    messages = state["messages"]
    question = messages[0].content
    msg = [
        HumanMessage(
            content=f"""
            Look at the input and try to reason about the underlying semantic intent / meaning.

            Here is the initial question:

            -------

            {question}

            -------

            Formulate an improved question: """
        )
    ]
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}

def generate(state):
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}
