import streamlit as st
from chat_botbe import chatbot
from langchain_core.messages import HumanMessage


# Thread ID
CONFIG = {
    "configurable": {
        "thread_id": "1"
    }
}


# Initialize message history
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Display previous messages
for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.text(message["content"])


# Chat input
user_input = st.chat_input("Type here...")


if user_input:

    # Display user message
    with st.chat_message("user"):
        st.text(user_input)

    # Save user message
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })


    # Send message to LangGraph
    response = chatbot.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=CONFIG
    )


    # Get AI response
    ai_message = response["messages"][-1].content

    # Gemini may return content as a list
    if isinstance(ai_message, list):
        ai_message = ai_message[0]["text"]


    # Save AI response
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })


    # Display AI response
    with st.chat_message("assistant"):
        st.text(ai_message)