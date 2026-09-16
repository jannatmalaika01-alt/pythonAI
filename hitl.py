from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt
from langchain_core.messages import SystemMessage,HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)
class human_state(TypedDict):
    approval:bool
    answer:str

def ai_work(state:human_state):
    messages=[
            SystemMessage(content="youre a mail generator llm"),
            HumanMessage(content=f'write an email to sara to tell her im late')
        ]
    response=llm.invoke(messages).content
    return {"answer":response}
    
def human_review(state:human_state):
    decision=interrupt(
        f"do you approve this answer? \n\n {state['answer']}"
    )
    return {"approval":decision}

def final_answer(state:human_state):
    if human_state["approval"]:
        return {"answer":state["answer"]}
    return {"answer":"answer rejected by human"}

graph=StateGraph(human_state)

graph.add_node("ai_work",ai_work)
graph.add_node("human_review",human_review)
graph.add_node("final_answer",final_answer)

graph.add_edge(START,"ai_work")
graph.add_edge("ai_work","human_review")
graph.add_edge("human_review","final_answer")
graph.add_edge("final_answer",END)

app=graph.compile()