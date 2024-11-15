import streamlit as st
from RAG_Chain import rag_chain
import time
from langchain.memory import ConversationBufferWindowMemory

def extract_result(response_text):
    # Locate the position of the word "assistant" and extract everything after it
    keyword = "assistant"
    position = response_text.find(keyword)
    
    # If "assistant" is found, extract the text that follows it
    if position != -1:
        # Extract everything after "assistant" and strip any leading/trailing whitespace
        extracted_text = response_text[position + len(keyword):].strip()
        return extracted_text
    else:
        return "No text found after 'assistant'."

# Streamlit UI setup
st.set_page_config(page_title="Vietnamese Legal ChatBot")
col1, col2, col3 = st.columns([1, 4, 1])
st.title("Vietnamese Legal ChatBot")
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ffd0d0;
    }
    div.stButton > button:active {
        background-color: #ff6262;
    }
    div[data-testid="stStatusWidget"] div button {
        display: none;
    }
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    button[title="View fullscreen"] {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Reset conversation function
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

# Input prompt
input_prompt = st.chat_input("Say something")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            result =  rag_chain.invoke({"query": input_prompt})
            message_placeholder = st.empty()
            full_response = "\n\n\n"

            # Print the result dictionary to inspect its structure
            #st.write(result)

            response = extract_result(result['result'])

            for chunk in response:
                full_response += chunk
                time.sleep(0.02)
                message_placeholder.markdown(full_response + " ▌")

            # Print the answer
            #st.write(result["result"])

        st.button('Reset All Chat 🗑️', on_click=reset_conversation)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
