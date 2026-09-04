import streamlit as st
import tempfile
import os
import uuid

import chat_botbe

from chat_botbe import (
    chatbot,
    retrieve_all_threads,
    save_chat_name,
    retrieve_all_chat_names,
    ingest_pdf
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Study Buddy",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def generate_thread_id():

    thread_id = str(
        uuid.uuid4()
    )

    return thread_id


# ============================================================
# ADD THREAD
# ============================================================

def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:

        st.session_state[
            "chat_threads"
        ].append(
            thread_id
        )


# ============================================================
# RESET CHAT
# ============================================================

def reset_chat():

    thread_id = generate_thread_id()


    st.session_state[
        "thread_id"
    ] = thread_id


    add_thread(
        thread_id
    )


    st.session_state[
        "message_history"
    ] = []


    # Reset PDF for new conversation

    st.session_state[
        "pdf_name"
    ] = None


    chat_botbe.retriever = None


# ============================================================
# LOAD CONVERSATION
# ============================================================

def load_conversation(thread_id):

    state = chatbot.get_state(

        config={
            "configurable": {
                "thread_id": thread_id
            }
        }

    )


    return state.values.get(
        "messages",
        []
    )


# ============================================================
# INITIALIZE MESSAGE HISTORY
# ============================================================

if "message_history" not in st.session_state:

    st.session_state[
        "message_history"
    ] = []


# ============================================================
# INITIALIZE THREAD ID
# ============================================================

if "thread_id" not in st.session_state:

    st.session_state[
        "thread_id"
    ] = generate_thread_id()


# ============================================================
# INITIALIZE THREADS
# ============================================================

if "chat_threads" not in st.session_state:

    st.session_state[
        "chat_threads"
    ] = retrieve_all_threads()


# ============================================================
# INITIALIZE CHAT NAMES
# ============================================================

if "chat_names" not in st.session_state:

    st.session_state[
        "chat_names"
    ] = retrieve_all_chat_names()


# ============================================================
# INITIALIZE PDF NAME
# ============================================================

if "pdf_name" not in st.session_state:

    st.session_state[
        "pdf_name"
    ] = None


# ============================================================
# ADD CURRENT THREAD
# ============================================================

add_thread(
    st.session_state["thread_id"]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Study Buddy"
)


# ============================================================
# NEW CHAT BUTTON
# ============================================================

if st.sidebar.button(
    "New Chat",
    use_container_width=True
):

    reset_chat()

    st.rerun()


# ============================================================
# CONVERSATIONS
# ============================================================

st.sidebar.header(
    "My Conversations"
)


# ============================================================
# NEWEST CHAT FIRST
# ============================================================

for thread_id in st.session_state[
    "chat_threads"
][::-1]:


    chat_name = st.session_state[
        "chat_names"
    ].get(
        thread_id,
        "New Chat"
    )


    if st.sidebar.button(

        chat_name,

        key=str(thread_id),

        use_container_width=True

    ):


        # Change thread

        st.session_state[
            "thread_id"
        ] = thread_id


        # Load messages

        messages = load_conversation(
            thread_id
        )


        temp_messages = []


        for message in messages:


            # Ignore tool messages

            if isinstance(
                message,
                ToolMessage
            ):

                continue


            # Determine role

            if isinstance(
                message,
                HumanMessage
            ):

                role = "user"


            elif isinstance(
                message,
                AIMessage
            ):

                role = "assistant"


            else:

                continue


            # Get content

            content = message.content


            # Gemini can sometimes return
            # a list instead of a string

            if isinstance(
                content,
                list
            ):

                text_content = ""


                for item in content:


                    if isinstance(
                        item,
                        dict
                    ):

                        if item.get(
                            "type"
                        ) == "text":

                            text_content += item.get(
                                "text",
                                ""
                            )


                    elif isinstance(
                        item,
                        str
                    ):

                        text_content += item


                content = text_content


            # Store message

            temp_messages.append({

                "role": role,

                "content": content

            })


        st.session_state[
            "message_history"
        ] = temp_messages


        st.rerun()


# ============================================================
# CURRENT THREAD CONFIG
# ============================================================

CONFIG = {

    "configurable": {

        "thread_id":
        st.session_state[
            "thread_id"
        ]

    }

}


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state[
    "message_history"
]:


    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ============================================================
# CHAT INPUT + PDF UPLOAD
# ============================================================

prompt = st.chat_input(

    "Ask anything...",

    accept_file=True,

    file_type=["pdf"]

)


# ============================================================
# IF USER SUBMITTED SOMETHING
# ============================================================

if prompt:


    # --------------------------------------------------------
    # CURRENT THREAD
    # --------------------------------------------------------

    current_thread = st.session_state[
        "thread_id"
    ]


    # --------------------------------------------------------
    # HANDLE PDF UPLOAD
    # --------------------------------------------------------

    if prompt.files:


        uploaded_file = prompt.files[0]


        # Store PDF name

        st.session_state[
            "pdf_name"
        ] = uploaded_file.name


        # Show upload status

        with st.spinner(
            "Processing PDF..."
        ):


            # Create temporary PDF file

            with tempfile.NamedTemporaryFile(

                delete=False,

                suffix=".pdf"

            ) as temp_file:


                temp_file.write(
                    uploaded_file.getbuffer()
                )


                temp_pdf_path = (
                    temp_file.name
                )


            try:


                # ------------------------------------------------
                # INGEST PDF
                # ------------------------------------------------

                vector_store = ingest_pdf(
                    temp_pdf_path
                )


                # ------------------------------------------------
                # CREATE RETRIEVER
                # ------------------------------------------------

                chat_botbe.retriever = (
                    vector_store.as_retriever(

                        search_type="similarity",

                        search_kwargs={
                            "k": 4
                        }

                    )
                )


                st.success(
                    f"PDF uploaded: {uploaded_file.name}"
                )


            except Exception as e:

                st.error(
                    f"Error processing PDF: {e}"
                )


            finally:


                # Delete temporary file

                if os.path.exists(
                    temp_pdf_path
                ):

                    os.remove(
                        temp_pdf_path
                    )


    # --------------------------------------------------------
    # GET TEXT FROM CHAT INPUT
    # --------------------------------------------------------

    current_input = prompt.text


    # --------------------------------------------------------
    # IF ONLY PDF WAS UPLOADED
    # --------------------------------------------------------

    if not current_input:

        st.stop()


    # --------------------------------------------------------
    # CREATE CHAT NAME
    # FROM FIRST MESSAGE
    # --------------------------------------------------------

    if current_thread not in st.session_state[
        "chat_names"
    ]:


        if len(current_input) > 30:

            chat_name = (
                current_input[:30]
                + "..."
            )

        else:

            chat_name = current_input


        # Save in session

        st.session_state[
            "chat_names"
        ][
            current_thread
        ] = chat_name


        # Save permanently

        save_chat_name(

            current_thread,

            chat_name

        )


    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.write(
            current_input
        )


    # --------------------------------------------------------
    # SAVE USER MESSAGE LOCALLY
    # --------------------------------------------------------

    st.session_state[
        "message_history"
    ].append({

        "role": "user",

        "content": current_input

    })


    # ========================================================
    # STREAM RESPONSE
    # ========================================================

    def stream_response(
        user_message
    ):


        for (
            message_chunk,
            metadata
        ) in chatbot.stream(

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


            # Only process AI messages

            if not isinstance(
                message_chunk,
                AIMessage
            ):

                continue


            content = (
                message_chunk.content
            )


            # ------------------------------------------------
            # GEMINI LIST RESPONSE
            # ------------------------------------------------

            if isinstance(
                content,
                list
            ):


                for item in content:


                    if isinstance(
                        item,
                        dict
                    ):


                        if item.get(
                            "type"
                        ) == "text":


                            text = item.get(
                                "text",
                                ""
                            )


                            if text:

                                yield text


                    elif isinstance(
                        item,
                        str
                    ):


                        if item:

                            yield item


            # ------------------------------------------------
            # NORMAL STRING RESPONSE
            # ------------------------------------------------

            elif isinstance(
                content,
                str
            ):


                if content:

                    yield content


    # ========================================================
    # DISPLAY AI RESPONSE
    # ========================================================

    with st.chat_message(
        "assistant"
    ):


        ai_response = st.write_stream(

            stream_response(
                current_input
            )

        )


    # ========================================================
    # SAVE AI RESPONSE
    # ========================================================

    st.session_state[
        "message_history"
    ].append({

        "role": "assistant",

        "content": ai_response

    })