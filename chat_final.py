import base64
import os
from itertools import count
import streamlit as st
import os
import types
from pathlib import Path
from collections.abc import Iterator
import requests

def get_answer(question: str) -> str:
    response = requests.post("http://127.0.0.1:8009/ask-question", json={"question": question})['answer']
    if response.status_code == 200:
        return response
    else:
        return "Error: Could not get the answer."

# Function to get base64 encoded string for image
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

developers = [
    "Paulo Ferreira"
]

# Main app function
def page_chat():
    develops_txt = [f"<a href='mailto:{dev}'>{dev}</a>" for dev in developers]
    sidebar_content = """
    **SQL Agent** 
    """
    st.sidebar.title(sidebar_content)

    option = st.sidebar.radio(
                            "Choose an option",
                            ('fine_tuned_model', 'base_model')
                            )


    choose_model = requests.post("http://127.0.0.1:8009/choose-model", json={"model_name": option})

    #st.sidebar.image('lablabai_logo.png')

    #st.sidebar.markdown('Developed by Generative Frontiers Lab<br>', unsafe_allow_html=True)

    # st.sidebar.title("Developed with:")
    # st.sidebar.image('vectara_logo.jpg')
    # st.sidebar.image('together_logo.jpg')
    st.title("Agents")

    # Initialize session state for selected options
    if "selected_options" not in st.session_state:
        st.session_state.selected_options = []

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        model_response = get_answer(prompt)

        with st.chat_message("assistant"):
            st.write(model_response)

        st.session_state.messages.append({"role": "assistant", "content": model_response})

if __name__ == '__main__':
    page_chat()

