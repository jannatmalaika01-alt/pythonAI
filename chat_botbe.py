from dotenv import load_dotenv
import os
import sqlite3

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


# Load .env file
load_dotenv()


# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Chat node
def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


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


# Add edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


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