import streamlit as st
from chat_botbe import (
    chatbot,
    retrieve_all_threads,
    save_chat_name,
    retrieve_all_chat_names
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid


#create utility functions
def generate_thread_id():
    thread_id = str(uuid.uuid4())
    return thread_id


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state['thread_id'] = thread_id

    add_thread(st.session_state['thread_id'])

    st.session_state['message_history'] = []


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):

    return chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    ).values.get('messages', [])


# Initialize message history
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()


if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()


if 'chat_names' not in st.session_state:
    st.session_state['chat_names'] = retrieve_all_chat_names()


add_thread(st.session_state['thread_id'])


#sidebar ui
st.sidebar.title("LangGraph Chatbot")


if st.sidebar.button("New Chat"):

    reset_chat()

    st.rerun()


st.sidebar.header("My Conversations")


# Show newest chat on top
for thread_id in st.session_state['chat_threads'][::-1]:

    chat_name = st.session_state['chat_names'].get(
        thread_id,
        "New Chat"
    )


    if st.sidebar.button(
        chat_name,
        key=str(thread_id)
    ):

        st.session_state['thread_id'] = thread_id


        messages = load_conversation(thread_id)


        temp_messages = []


        for message in messages:

            # Ignore tool messages
            if isinstance(message, ToolMessage):
                continue


            if isinstance(message, HumanMessage):

                role = "user"

            elif isinstance(message, AIMessage):

                role = "assistant"

            else:

                continue


            content = message.content


            # Gemini sometimes returns a list
            if isinstance(content, list):

                text_content = ""

                for item in content:

                    if isinstance(item, dict):

                        if item.get("type") == "text":

                            text_content += item.get(
                                "text",
                                ""
                            )

                    elif isinstance(item, str):

                        text_content += item


                content = text_content


            temp_messages.append({
                'role': role,
                'content': content
            })


        st.session_state['message_history'] = temp_messages

        st.rerun()


# Thread ID
CONFIG = {
    "configurable": {
        "thread_id": st.session_state['thread_id']
    }
}


# Display previous messages
for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# Chat input
user_input = st.chat_input("Type here...")


if user_input:

    # Store user input immediately
    current_input = user_input


    # Current thread
    current_thread = st.session_state['thread_id']


    # Create chat name from first message
    if current_thread not in st.session_state['chat_names']:

        if len(current_input) > 30:

            chat_name = current_input[:30] + "..."

        else:

            chat_name = current_input


        # Store in session state
        st.session_state['chat_names'][current_thread] = chat_name


        # Store permanently in SQLite
        save_chat_name(
            current_thread,
            chat_name
        )


    # Display user message
    with st.chat_message("user"):

        st.write(current_input)


    # Save user message
    st.session_state["message_history"].append({
        "role": "user",
        "content": current_input
    })


    # Function for streaming
    def stream_response(user_message):

        for message_chunk, metadata in chatbot.stream(

            {
                "messages": [
                    HumanMessage(
                        content=user_message
                    )
                ]
            },

            config=CONFIG,

            stream_mode="messages"
        ):

            # Only stream AI messages
            if not isinstance(message_chunk, AIMessage):
                continue


            content = message_chunk.content


            # Gemini sometimes returns a list
            if isinstance(content, list):

                for item in content:

                    if isinstance(item, dict):

                        if item.get("type") == "text":

                            text = item.get(
                                "text",
                                ""
                            )

                            if text:

                                yield text


                    elif isinstance(item, str):

                        yield item


            # Normal string content
            elif isinstance(content, str):

                if content:

                    yield content


    # Display streaming AI response
    with st.chat_message("assistant"):

        ai_response = st.write_stream(
            stream_response(current_input)
        )


    # Save AI response
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_response
    })