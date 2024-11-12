# Import required libraries
import streamlit as st
import random
import time
import numpy as np
from dotenv import load_dotenv
from itertools import zip_longest
from streamlit_chat import message
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from hume.client import AsyncHumeClient
from hume.empathic_voice.chat.socket_client import ChatConnectOptions, ChatWebsocketConnection
from hume.empathic_voice.chat.types import SubscribeEvent
from hume.empathic_voice.types import UserInput
from hume.core.api_error import ApiError
from hume import MicrophoneInterface, Stream
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

personality = """You are a helpful and experienced interpersonal skills coach called Eque (pronounced e-que) who is talking with a human. 

        Your job is to help humans who talk with you identify their strengths and limitations when it comes to their interpersonal skills.
        
        You pay careful attention to the tone of the person speaking to you, and you reflect back to them what you think they might be thinking.

        You are an expert in things like active listening, collaboration, creative thinking, communication, influencing and self reflection.
        
        You then ask them which interpersonal skills they might like to work on - things like active listening, clear communication, influencing, creative thinking, and more.
        
        You should guide the conversation in a way so that you can determine what the weaknesses are in these areas in the person talking to you, and then give them specific techniques to improve on these weaknesses. 
        
        Test them using examples and try to get them to use the techniques on you to practice. 
        
        Give them constructive feedback, and tell them what they have done well in as well. 
        
        At the start of the call or chat ask them what they would like help with out of the list I gave you. 
        
        Make sure the person has finished talking before you jump in - you don't want to interrupt them!"""

# Load environment variables
load_dotenv()

# Set streamlit page configuration
st.set_page_config(page_title="Human Skills Coach")

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("eque.png")

st.title("""Your personal *human skills* coach.""")
st.markdown("""As AI handles the technical day-to-day, human skills matter more than ever. 
            Practice your interpersonal skills with Eque to nail your next interview, lead confidently, 
            or communicate more effectively. No pressure – just guided practice via text (voice coming soon).
            Build the interpersonal skills that set you apart.
    """
    , unsafe_allow_html=False, help=None)


# Initialize the ChatOpenAI model
llm = ChatOpenAI(temperature=0.5, model_name="gpt-4o", streaming = True)

# Specify session_state key for storing messages
msgs = StreamlitChatMessageHistory(key="special_app_key")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", personality),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,  # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="history",
)

for msg in msgs.messages:
    if msg.type == "human":
        st.chat_message(msg.type, avatar = "🧑‍💻").write(msg.content)
    elif msg.type == "ai":
        st.chat_message(msg.type, avatar = "🤖").write(msg.content)

if prompt := st.chat_input("Ask me what I can help you with!"):
    st.chat_message("human", avatar = "🧑‍💻").write(prompt)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai", avatar = "🤖").write(response.content)


# Add credit
with st.sidebar:
    st.markdown('''Made with 💖 by Ash Stevens  
    https://github.com/ashjstevens/human-skills
    ''')