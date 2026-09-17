from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    decision = interrupt({
        "type": "approval",
        "reason": "Model is about to answer a human question.",
        "question": state["messages"][-1].content,
        "instruction": "Do you approve this message?"
    })

    if decision["approved"] == "no":
        return {"messages": [AIMessage(content="Not approved.")]}
    else:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

builder = StateGraph(ChatState)

builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ResumeRequest(BaseModel):
    thread_id: str
    approved: str

@app.post("/chat")
def chat(request: ChatRequest):

    config = {
        "configurable": {
            "thread_id": request.thread_id
        }
    }

    result = graph.invoke(
        {
            "messages": [
                ("user", request.message)
            ]
        },
        config=config
    )

    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value

        return {
            "status": "approval_required",
            "thread_id": request.thread_id,
            "message": interrupt_data
        }

    return {
        "status": "completed",
        "response": result["messages"][-1].content
    }

@app.post("/chat/resume")
def resume_chat(request: ResumeRequest):

    config = {
        "configurable": {
            "thread_id": request.thread_id
        }
    }

    result = graph.invoke(
        Command(
            resume={
                "approved": request.approved
            }
        ),
        config=config
    )

    return {
        "status": "completed",
        "response": result["messages"][-1].content
    }