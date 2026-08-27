from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)
#create states
class llmstate(TypedDict):
    question:str
    answer:str
#create function
def llm_qa(state:llmstate)->llmstate:
    #extract question
    question=state["question"]
    #form a prompt 
    prompt=f"Answer the following question:{question}"
    #ask the question 
    answer=llm.invoke(prompt).content
    #update the answer in the state
    state["answer"]=answer
    return state 
#create graph
graph = StateGraph(llmstate)
#add nodes
graph.add_node("llm_qa", llm_qa)
#add edges
graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)
#compile
workflow = graph.compile()
#execute
result=workflow.invoke({
    "question":"what is ai?",
    "answer":""
})
print(result)