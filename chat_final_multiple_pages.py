import streamlit as st

def intro():
    import streamlit as st

    st.write("# Welcome to Streamlit! 👋")
    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.

        **👈 Select a demo from the dropdown on the left** to see some examples
        of what Streamlit can do!

        ### Want to learn more?

        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)

        ### See more complex demos

        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )

import requests
def get_answer_langchain_sql_agent(question: str) -> str:
    response = requests.post("http://127.0.0.1:8000/ask_langchain_sql_agent", json={"text": question})
    if response.status_code == 200:
        return response.json()["text"]
    else:
        return "Error: Could not get the answer."
    

def ask_question_sql_coder(question: str) -> str:
    response = requests.post("http://127.0.0.1:8000/ask_sql_coder", json={"text": question})
    if response.status_code == 200:
        return response.json()["text"]
    else:
        return "Error: Could not get the answer."

def langchain_sql_agent():
    import streamlit as st

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This demo uses the create_sql_agent from langchain toolkit
"""
    )
    
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

        model_response = get_answer_langchain_sql_agent(prompt)

        with st.chat_message("assistant"):
            st.write(model_response)

        st.session_state.messages.append({"role": "assistant", "content": model_response})

def sql_coder():
    import streamlit as st

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This demo uses the create_sql_agent from langchain toolkit
"""
    )
    
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

        model_response = ask_question_sql_coder(prompt)

        with st.chat_message("assistant"):
            st.write(model_response)

        st.session_state.messages.append({"role": "assistant", "content": model_response})

page_names_to_funcs = {
    "—": intro,
    "SQL Coder": sql_coder,
    "LangChain SQL Agent": langchain_sql_agent,
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()