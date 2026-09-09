import streamlit as st
import tempfile
import os
import uuid
import base64

import chat_botbe

from chat_botbe import (
    chatbot,
    retrieve_all_threads,
    save_chat_name,
    retrieve_all_chat_names,
    ingest_pdf,
    pin_chat,
    unpin_chat,
    is_chat_pinned,
    delete_chat,
    rename_chat,
    touch_thread
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage
)

st.set_page_config(
    page_title="Study Buddy",
    page_icon="📚",
    layout="wide"
)

def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["pdf_name"] = None
    st.session_state["uploaded_image"] = None

    chat_botbe.retriever = None

    add_thread(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])

def refresh_threads():
    st.session_state["chat_threads"] = retrieve_all_threads()
    st.session_state["chat_names"] = retrieve_all_chat_names()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "chat_names" not in st.session_state:
    st.session_state["chat_names"] = retrieve_all_chat_names()

if "pdf_name" not in st.session_state:
    st.session_state["pdf_name"] = None

if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None

add_thread(st.session_state["thread_id"])

st.sidebar.title("📚 Study Buddy")

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):
    reset_chat()
    st.rerun()

st.sidebar.divider()

st.sidebar.header("My Conversations")

pinned_threads = []
recent_threads = []

for thread_id in st.session_state["chat_threads"]:

    if is_chat_pinned(thread_id):
        pinned_threads.append(thread_id)
    else:
        recent_threads.append(thread_id)

if pinned_threads:

    st.sidebar.subheader("📌 Pinned")

    for thread_id in pinned_threads:

        chat_name = st.session_state["chat_names"].get(
            thread_id,
            "New Chat"
        )

        col1, col2 = st.sidebar.columns(
            [5, 1]
        )

        with col1:

            if st.button(
                chat_name,
                key=f"chat_{thread_id}",
                use_container_width=True
            ):

                st.session_state["thread_id"] = thread_id

                messages = load_conversation(thread_id)

                temp_messages = []

                for message in messages:

                    if isinstance(message, ToolMessage):
                        continue

                    if isinstance(message, HumanMessage):
                        role = "user"

                    elif isinstance(message, AIMessage):
                        role = "assistant"

                    else:
                        continue

                    content = message.content

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
                        "role": role,
                        "content": content
                    })

                st.session_state["message_history"] = temp_messages
                st.session_state["uploaded_image"] = None

                st.rerun()

        with col2:

            if st.button(
                "⋮",
                key=f"menu_{thread_id}"
            ):
                st.session_state[
                    f"menu_open_{thread_id}"
                ] = not st.session_state.get(
                    f"menu_open_{thread_id}",
                    False
                )

        if st.session_state.get(
            f"menu_open_{thread_id}",
            False
        ):

            if st.sidebar.button(
                "📌 Unpin",
                key=f"unpin_{thread_id}",
                use_container_width=True
            ):

                unpin_chat(thread_id)
                refresh_threads()
                st.rerun()

            if st.sidebar.button(
                "✏️ Rename",
                key=f"rename_{thread_id}",
                use_container_width=True
            ):

                st.session_state[
                    f"rename_mode_{thread_id}"
                ] = True

            if st.sidebar.button(
                "🗑️ Delete",
                key=f"delete_{thread_id}",
                use_container_width=True
            ):

                delete_chat(thread_id)

                if st.session_state["thread_id"] == thread_id:
                    reset_chat()

                refresh_threads()
                st.rerun()

            if st.session_state.get(
                f"rename_mode_{thread_id}",
                False
            ):

                new_name = st.sidebar.text_input(
                    "New name",
                    value=chat_name,
                    key=f"rename_input_{thread_id}"
                )

                if st.sidebar.button(
                    "Save",
                    key=f"save_rename_{thread_id}",
                    use_container_width=True
                ):

                    rename_chat(
                        thread_id,
                        new_name
                    )

                    st.session_state[
                        f"rename_mode_{thread_id}"
                    ] = False

                    refresh_threads()
                    st.rerun()

for thread_id in recent_threads:

    chat_name = st.session_state["chat_names"].get(
        thread_id,
        "New Chat"
    )

    col1, col2 = st.sidebar.columns(
        [5, 1]
    )

    with col1:

        if st.button(
            chat_name,
            key=f"chat_{thread_id}",
            use_container_width=True
        ):

            st.session_state["thread_id"] = thread_id

            messages = load_conversation(thread_id)

            temp_messages = []

            for message in messages:

                if isinstance(message, ToolMessage):
                    continue

                if isinstance(message, HumanMessage):
                    role = "user"

                elif isinstance(message, AIMessage):
                    role = "assistant"

                else:
                    continue

                content = message.content

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
                    "role": role,
                    "content": content
                })

            st.session_state["message_history"] = temp_messages
            st.session_state["uploaded_image"] = None

            st.rerun()

    with col2:

        if st.button(
            "⋮",
            key=f"menu_{thread_id}"
        ):

            st.session_state[
                f"menu_open_{thread_id}"
            ] = not st.session_state.get(
                f"menu_open_{thread_id}",
                False
            )

    if st.session_state.get(
        f"menu_open_{thread_id}",
        False
    ):

        if st.sidebar.button(
            "📌 Pin",
            key=f"pin_{thread_id}",
            use_container_width=True
        ):

            pin_chat(thread_id)
            refresh_threads()
            st.rerun()

        if st.sidebar.button(
            "✏️ Rename",
            key=f"rename_{thread_id}",
            use_container_width=True
        ):

            st.session_state[
                f"rename_mode_{thread_id}"
            ] = True

        if st.sidebar.button(
            "🗑️ Delete",
            key=f"delete_{thread_id}",
            use_container_width=True
        ):

            delete_chat(thread_id)

            if st.session_state["thread_id"] == thread_id:
                reset_chat()

            refresh_threads()
            st.rerun()

        if st.session_state.get(
            f"rename_mode_{thread_id}",
            False
        ):

            new_name = st.sidebar.text_input(
                "New name",
                value=chat_name,
                key=f"rename_input_{thread_id}"
            )

            if st.sidebar.button(
                "Save",
                key=f"save_rename_{thread_id}",
                use_container_width=True
            ):

                rename_chat(
                    thread_id,
                    new_name
                )

                st.session_state[
                    f"rename_mode_{thread_id}"
                ] = False

                refresh_threads()
                st.rerun()

CONFIG = {
    "configurable": {
        "thread_id": st.session_state["thread_id"]
    }
}

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input(
    "Ask anything...",
    accept_file=True,
    file_type=[
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "webp"
    ]
)

if prompt:

    current_thread = st.session_state["thread_id"]

    if prompt.files:

        uploaded_file = prompt.files[0]

        if uploaded_file.type == "application/pdf":

            st.session_state["pdf_name"] = uploaded_file.name

            with st.spinner(
                "Processing PDF..."
            ):

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_pdf_path = temp_file.name

                try:

                    vector_store = ingest_pdf(
                        temp_pdf_path
                    )

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

                    if os.path.exists(
                        temp_pdf_path
                    ):
                        os.remove(
                            temp_pdf_path
                        )

        elif uploaded_file.type.startswith(
            "image/"
        ):

            st.session_state[
                "uploaded_image"
            ] = uploaded_file

    current_input = prompt.text

    if not current_input:

        if st.session_state[
            "uploaded_image"
        ] is not None:

            st.info(
                "Image attached. Ask a question about it."
            )

        else:
            st.stop()

    if current_thread not in st.session_state[
        "chat_names"
    ]:

        if len(current_input) > 30:

            chat_name = (
                current_input[:30] + "..."
            )

        else:
            chat_name = current_input

        st.session_state[
            "chat_names"
        ][current_thread] = chat_name

        save_chat_name(
            current_thread,
            chat_name
        )

    touch_thread(current_thread)

    uploaded_image = st.session_state[
        "uploaded_image"
    ]

    if uploaded_image is not None:

        image_bytes = uploaded_image.getvalue()

        image_base64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        image_message = [
            {
                "type": "text",
                "text": current_input
            },
            {
                "type": "image_url",
                "image_url": {
                    "url":
                        f"data:{uploaded_image.type};base64,{image_base64}"
                }
            }
        ]

    else:

        image_message = current_input

    with st.chat_message("user"):

        if uploaded_image is not None:
            st.image(uploaded_image)

        st.write(current_input)

    st.session_state[
        "message_history"
    ].append({
        "role": "user",
        "content": current_input
    })

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

            if not isinstance(
                message_chunk,
                AIMessage
            ):
                continue

            content = message_chunk.content

            if isinstance(content, list):

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

            elif isinstance(
                content,
                str
            ):

                if content:
                    yield content

    with st.chat_message("assistant"):

        ai_response = st.write_stream(
            stream_response(
                image_message
            )
        )

    st.session_state[
        "message_history"
    ].append({
        "role": "assistant",
        "content": ai_response
    })

    touch_thread(current_thread)

    st.session_state[
        "uploaded_image"
    ] = None

    st.session_state[
        "chat_threads"
    ] = retrieve_all_threads()