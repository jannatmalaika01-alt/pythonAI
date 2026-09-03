from dotenv import load_dotenv
import os
import sqlite3

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import requests
import random 
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
# Load .env file
load_dotenv()


# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
# load the pdf
loader=PyPDFLoader("wed servlets.pdf")
docs=loader.load()
#get its length
len(docs)
#now we split the doc
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitter.split_documents(docs)
#now we get length of chunk 
len(chunks)
#gemini embedding class
embeddings=GoogleGenerativeAIEmbeddings( model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"))
# now faiss db 
vector_store=FAISS.from_documents(chunks,embeddings)
vector_store
#create retriever 
retriever =vector_store.as_retriever(search_type='similarity',search_kwargs={'k':4})
# making a rag retriebver tool
@tool
def rag_retriever(query):
    """Retreive relevant information from the pdf document . Use this tool when the user asks facual /conceptual questions that might be answered from the stored document."""
    result=retriever.invoke(query)
    context=[doc.page_content for doc in result]
    meta_data=[doc.metadata for doc in result]
    return{
        'query':query,
        'context':context,
        'meta_data':meta_data
    }




searchTool=DuckDuckGoSearchRun(region='us-en')
@tool
def calculator(first_number: float, second_number: float, operation: str) -> dict:
    """ perform basic arithmetic operations on two given  numbers,allowed operations are add, subtract,multiply, and divide. """
    if operation == "add":
        result= first_number + second_number
    elif operation == "subtract":
        result= first_number - second_number
    elif operation == "multiply":
        result= first_number * second_number
    elif operation == "divide":
        if second_number != 0:
            result= first_number / second_number
        else:
            raise ValueError("Cannot divide by zero.")
    else:
        raise ValueError("Invalid operation. Supported operations: add, subtract, multiply, divide.")
    return {'first_number': first_number, 'second_number': second_number, 'operation': operation, 'result': result}
# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@tool
def get_stock_price(symbol:str)->str:
    """get the current stock price for a given stock symbol."""
    url=(  f"https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={ALPHA_VANTAGE_API_KEY}"
    )

    r=requests.get(url)
    return r.json()
tools=[get_stock_price,calculator,searchTool]   
llm_with_tools=llm.bind_tools(tools)
# Chat node
def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }
tool_node=ToolNode(tools)

# SQLite connection
conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)


# LangGraph memory
checkpointer = SqliteSaver(conn=conn)


# Create separate table for chat names
conn.execute("""
    CREATE TABLE IF NOT EXISTS chat_names (
        thread_id TEXT PRIMARY KEY,
        chat_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

conn.commit()


# Create graph
graph = StateGraph(ChatState)


# Add node
graph.add_node("chat_node", chat_node)
graph.add_node("tools",tool_node)

# Add edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")
# Compile chatbot
chatbot = graph.compile(
    checkpointer=checkpointer
)


# Get all threads
def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        thread_id = checkpoint.config["configurable"]["thread_id"]

        all_threads.add(thread_id)

    return list(all_threads)


# Save chat name
def save_chat_name(thread_id, chat_name):

    conn.execute(
        """
        INSERT OR REPLACE INTO chat_names (thread_id, chat_name)
        VALUES (?, ?)
        """,
        (str(thread_id), chat_name)
    )

    conn.commit()


# Get chat name
def get_chat_name(thread_id):

    cursor = conn.execute(
        """
        SELECT chat_name
        FROM chat_names
        WHERE thread_id = ?
        """,
        (str(thread_id),)
    )

    result = cursor.fetchone()

    if result:
        return result[0]

    return "New Chat"


# Get all chat names
def retrieve_all_chat_names():

    chat_names = {}

    cursor = conn.execute(
        """
        SELECT thread_id, chat_name
        FROM chat_names
        """
    )

    rows = cursor.fetchall()

    for thread_id, chat_name in rows:

        chat_names[thread_id] = chat_name

    return chat_names