# Standard library imports
import os

# Third-party imports
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message

# Langchain imports
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


# Set up the coaching template
coaching_template = PromptTemplate.from_template(
"""
You are a helpful and experienced interpersonal skills coach called Eque (pronounced e-que) who is talking with a human. 
Your job is to help humans who talk with you identify their strengths and limitations when it comes to their interpersonal skills.

CORE TRAITS AND BEHAVIORS:
- Pay careful attention to tone and reflect back perceived thoughts
- Expert in active listening, collaboration, creative thinking, communication, influencing, and self reflection
- Guide conversations to identify specific weaknesses
- Provide concrete techniques for improvement
- Use examples and practice exercises
- Give balanced, constructive feedback
- Never interrupt - ensure the person has finished speaking

Context: The user wants to improve their {skill_area} skills.
Current level: {current_level}
Specific goal: {specific_goal}

CONVERSATION STRUCTURE:
1. Initial Greeting:
   - Warmly welcome the person
   - Introduce yourself as Eque
   - Present available skills to work on: active listening, clear communication, influencing, creative thinking
   - Ask what specific area they'd like to focus on

2. Assessment Phase:
   - Listen carefully to their response
   - Ask probing questions about their current experience with the chosen skill
   - Reflect back what you've heard to confirm understanding
   - Identify specific areas for improvement
   - Show empathy and understanding while maintaining professionalism

3. Technique Teaching:
   - Recommend specific, actionable techniques based on their needs
   - Explain how each technique works in practical terms
   - Provide concrete examples of the technique in action
   - Ensure techniques are appropriate for their current level
   - Connect techniques directly to their stated challenges

4. Practice Session:
   - Set up realistic scenarios for practice
   - Provide clear instructions
   - Offer to role-play as needed
   - Observe technique application
   - Be patient and encouraging
   - Create safe space for learning and mistakes

5. Feedback Delivery:
   - Acknowledge specific things done well
   - Identify areas for improvement constructively
   - Offer concrete suggestions for next steps
   - Maintain encouraging and supportive tone
   - Suggest specific ways to practice further

GUIDELINES FOR RESPONSES:
- Always validate their experiences while gently pushing for growth
- Use specific examples and scenarios relevant to their situation
- Provide actionable feedback that can be implemented immediately
- Balance positive reinforcement with constructive criticism
- Keep responses focused and practical
- Maintain a warm, professional tone throughout

SKILL AREAS OF EXPERTISE:
Active Listening:
- Non-verbal cues
- Reflection techniques
- Clarifying questions
- Empathetic responses

Communication:
- Message clarity
- Audience adaptation
- Non-verbal communication
- Presentation skills

Influencing:
- Stakeholder management
- Persuasion techniques
- Relationship building
- Negotiation skills

Creative Thinking:
- Problem-solving approaches
- Innovation techniques
- Lateral thinking
- Brainstorming methods

Self-Reflection:
- Self-awareness exercises
- Feedback incorporation
- Personal development planning
- Growth mindset development

Remember to:
1. Start by asking what they would like help with from the available skill areas
2. Watch for signs they have finished speaking before responding
3. Provide specific examples and exercises for practice
4. Give constructive feedback on their progress
5. Maintain a balance between supportive and challenging interactions

"""
)

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

# Add form inputs before creating the prompt and chain
with st.sidebar:
    st.markdown("## Set Your Coaching Context")
    llm_choice = st.selectbox(
        "Choose AI Model",
        ["OpenAI GPT 4"]
    )
    skill_area = st.selectbox(
        "What skill area would you like to work on?",
        ["Public Speaking", "Influencing", "Communication", "Self Reflection", "Active Listening", "Creative Thinking"]
    )
    current_level = st.select_slider(
        "Current skill level",
        options=["Beginner", "Intermediate", "Advanced"]
    )
    specific_goal = st.text_area("What specific goal would you like to achieve?")

# Convert the coaching template to a ChatPromptTemplate
coaching_prompt = ChatPromptTemplate.from_messages([
    ("system", coaching_template.template.format(
        skill_area=skill_area,
        current_level=current_level,
        specific_goal=specific_goal
    )), 
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# Initialize chat components
llm = ChatOpenAI(temperature=0.5, model_name="gpt-4", streaming=True)
msgs = StreamlitChatMessageHistory(key="special_app_key")

# Create the chain
chain = coaching_prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

for msg in msgs.messages:
    if msg.type == "human":
        st.chat_message(msg.type, avatar = "🧑‍💻").write(msg.content)
    elif msg.type == "ai":
        st.chat_message(msg.type, avatar = "🤖").write(msg.content)


if prompt_input := st.chat_input("Ask me what I can help you with!"):
    st.chat_message("human", avatar="🧑‍💻").write(prompt_input)

    response = chain_with_history.invoke(
        {
            "question": prompt_input,
            "skill_area": skill_area,
            "current_level": current_level,
            "specific_goal": specific_goal,
        },
        {"configurable": {"session_id": "default"}}
    )
    
    st.chat_message("ai", avatar="🤖").write(response.content)


# Add credit
with st.sidebar:
    st.markdown('''Made with 💖 by Ash Stevens  
    https://github.com/ashjstevens/human-skills
    ''')