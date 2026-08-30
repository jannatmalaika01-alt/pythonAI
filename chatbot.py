# creating a chatbot
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
#ye ram ma store krega 
#the add msg is a built in to keep message history kinda..list ma msg ko append krega
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    #take user query
    messages=state["messages"]

    #send to llm
    response=llm.invoke(messages)

    #store response
    return {"messages":[response]}
checkpointer=MemorySaver()
#define graph
graph=StateGraph(ChatState)

#add nodes
graph.add_node("chat_node",chat_node)

#add edges
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

chatbot=graph.compile(checkpointer=checkpointer)

"""initial_state={
    "messages":[HumanMessage(content="whats the capital of pakistan")]
}

print(chatbot.invoke(initial_state)["messages"][-1].content)
"""
thread_id="1"
while True:
    user_message=input("Type here: ")
    print('user:' ,user_message)
    if user_message.strip().lower() in ['bye','exit','quit']:
        break
    # invoke ki wjah se har bar nai call or sirf nya msg list ma so we'll use persistance , as invoke does from scratch
    config={'configurable':{'thread_id':thread_id}}
    response=chatbot.invoke({'messages':[HumanMessage(content=user_message)]},config=config)
    print("AI: ",response['messages'][-1].content[0]["text"])