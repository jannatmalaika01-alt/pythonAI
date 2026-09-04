from dotenv import load_dotenv
import os
import sqlite3
import requests
import random

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools import DuckDuckGoSearchRun

from langchain_core.tools import tool

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


# ============================================================
# GEMINI LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GEMINI_API_KEY
)


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)


# ============================================================
# PDF INGESTION FUNCTION
# ============================================================

def ingest_pdf(pdf_path):
    """
    Load a PDF, split it into chunks, create Gemini embeddings,
    and store the chunks in a FAISS vector database.

    Returns:
        FAISS vector store
    """

    # Load the PDF
    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    print("PDF loaded successfully.")
    print("Number of pages:", len(docs))

    # Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    print("Document split successfully.")
    print("Number of chunks:", len(chunks))

    # Create FAISS vector store
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    print("FAISS vector store created successfully.")

    return vector_store


# ============================================================
# INGEST PDF
# ============================================================

vector_store = ingest_pdf("web servlets.pdf")


# ============================================================
# CREATE RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)


# ============================================================
# RAG RETRIEVER TOOL
# ============================================================

@tool
def rag_retriever(query):
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual
    questions that might be answered from the stored document.
    """

    result = retriever.invoke(query)

    context = [
        doc.page_content
        for doc in result
    ]

    meta_data = [
        doc.metadata
        for doc in result
    ]

    return {
        "query": query,
        "context": context,
        "meta_data": meta_data
    }


# ============================================================
# WEB SEARCH TOOL
# ============================================================

searchTool = DuckDuckGoSearchRun(
    region="us-en"
)


# ============================================================
# CALCULATOR TOOL
# ============================================================

@tool
def calculator(
    first_number: float,
    second_number: float,
    operation: str
) -> dict:
    """
    Perform basic arithmetic operations on two given numbers.

    Allowed operations:
    add, subtract, multiply, divide
    """

    if operation == "add":

        result = first_number + second_number

    elif operation == "subtract":

        result = first_number - second_number

    elif operation == "multiply":

        result = first_number * second_number

    elif operation == "divide":

        if second_number != 0:
            result = first_number / second_number
        else:
            raise ValueError("Cannot divide by zero.")

    else:

        raise ValueError(
            "Invalid operation. Supported operations: "
            "add, subtract, multiply, divide."
        )

    return {
        "first_number": first_number,
        "second_number": second_number,
        "operation": operation,
        "result": result
    }


# ============================================================
# CHAT STATE
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# STOCK PRICE TOOL
# ============================================================

@tool
def get_stock_price(symbol: str) -> str:
    """
    Get the current stock price for a given stock symbol.
    """

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={ALPHA_VANTAGE_API_KEY}"
    )

    r = requests.get(url)

    return r.json()


# ============================================================
# TOOLS
# ============================================================

tools = [
    get_stock_price,
    calculator,
    searchTool,
    rag_retriever
]


# ============================================================
# BIND TOOLS TO GEMINI
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# CHAT NODE
# ============================================================

def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# ============================================================
# TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# SQLITE CONNECTION
# ============================================================

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)


# ============================================================
# LANGGRAPH MEMORY
# ============================================================

checkpointer = SqliteSaver(
    conn=conn
)


# ============================================================
# CHAT NAMES TABLE
# ============================================================

conn.execute("""
    CREATE TABLE IF NOT EXISTS chat_names (
        thread_id TEXT PRIMARY KEY,
        chat_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

conn.commit()


# ============================================================
# CREATE GRAPH
# ============================================================

graph = StateGraph(ChatState)


# ============================================================
# ADD NODES
# ============================================================

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    tool_node
)


# ============================================================
# GRAPH EDGES
# ============================================================

graph.add_edge(
    START,
    "chat_node"
)

# IMPORTANT:
# Do not add chat_node -> END here.
#
# tools_condition decides whether:
#
# chat_node -> tools
#
# OR
#
# chat_node -> END

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge(
    "tools",
    "chat_node"
)


# ============================================================
# COMPILE CHATBOT
# ============================================================

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# GET ALL THREADS
# ============================================================

def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        thread_id = checkpoint.config[
            "configurable"
        ]["thread_id"]

        all_threads.add(thread_id)

    return list(all_threads)


# ============================================================
# SAVE CHAT NAME
# ============================================================

def save_chat_name(thread_id, chat_name):

    conn.execute(
        """
        INSERT OR REPLACE INTO chat_names
        (thread_id, chat_name)
        VALUES (?, ?)
        """,
        (
            str(thread_id),
            chat_name
        )
    )

    conn.commit()


# ============================================================
# GET CHAT NAME
# ============================================================

def get_chat_name(thread_id):

    cursor = conn.execute(
        """
        SELECT chat_name
        FROM chat_names
        WHERE thread_id = ?
        """,
        (
            str(thread_id),
        )
    )

    result = cursor.fetchone()

    if result:

        return result[0]

    return "New Chat"


# ============================================================
# GET ALL CHAT NAMES
# ============================================================

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