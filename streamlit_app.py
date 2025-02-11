# Standard library imports
import os
from typing import List, Dict

# Third-party imports
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message

# Langchain imports
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    Document
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory

# Initialize environment variables
load_dotenv()


# Set up the coaching template as a ChatPromptTemplate
coaching_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful and experienced interpersonal skills coach called Eque (pronounced e-que) who is talking with a human.
Your responses should be grounded in the following verified coaching methodologies and research:

IMPORTANT: Only give a welcome message if this is your first response in the conversation.

CORE TRAITS AND BEHAVIORS:
- Ground your responses in evidence-based coaching methodologies
- Pay careful attention to tone and reflect back perceived thoughts
- Expert in active listening, collaboration, creative thinking, communication, influencing, and self reflection
- Guide conversations to identify specific weaknesses
- Provide concrete techniques for improvement backed by research
- Use examples and practice exercises
- Give balanced, constructive feedback
- Never interrupt - ensure the person has finished speaking
     
CONVERSATION STRUCTURE:
1. Initial Greeting:
   - Warmly welcome the person only on the first message
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
1. Base your advice on verified coaching methodologies
2. Start by asking what they would like help with from the available skill areas
3. Watch for signs they have finished speaking before responding
4. Provide specific examples and exercises for practice
5. Give constructive feedback on their progress
6. Maintain a balance between supportive and challenging interactions
7. Cite specific research or methodologies when appropriate

"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


# Constants
COACHING_METHODOLOGIES = {
    "ICF Core Competencies": """
    Foundation:
    The International Coach Federation (ICF) defines coaching as partnering with clients in a thought-provoking and creative process that inspires them to maximize their personal and professional potential.

    Core Competencies Framework:
    1. Demonstrates Ethical Practice
    - Understands and consistently applies coaching ethics and standards
    - Maintains confidentiality with client information
    - Distinguishes between coaching, consulting, psychotherapy, and other support professions
    - Refers clients to other support professionals when needed

    2. Embodies a Coaching Mindset
    - Acknowledges that clients are responsible for their own choices
    - Engages in ongoing learning and development as a coach
    - Develops and maintains a reflective practice to enhance self-awareness
    - Remains aware of and open to the influence of context and culture on self and others

    3. Establishes and Maintains Agreements
    - Explains what is and is not appropriate in the coaching relationship
    - Reaches agreement about what is appropriate in the relationship and what is not
    - Reaches agreement about the coaching process and relationship
    - Reaches agreement about the parameters of confidentiality

    4. Cultivates Trust and Safety
    - Seeks to understand the client within their context
    - Demonstrates respect for the client's identity, perceptions, style, and language
    - Demonstrates openness and transparency as a way of displaying vulnerability and building trust
    """,

    "GROW Model": """
    Overview:
    The GROW Model is a simple yet powerful framework for structuring coaching sessions. Developed by Sir John Whitmore, it provides a clear process for goal setting and problem solving.

    Components:
    1. Goal
    - Establish specific, measurable goals
    - Ensure goals are achievable and relevant
    - Set time-bound objectives
    Key Questions:
    - "What do you want to achieve?"
    - "What would success look like?"
    - "How will you know you've achieved it?"

    2. Reality
    - Assess current situation
    - Identify and explore obstacles
    - Examine available resources
    Key Questions:
    - "What's happening now?"
    - "What steps have you taken so far?"
    - "What obstacles are you facing?"

    3. Options
    - Explore possible strategies
    - Generate alternative approaches
    - Evaluate pros and cons
    Key Questions:
    - "What alternatives are available?"
    - "What has worked in similar situations?"
    - "What would happen if you did nothing?"

    4. Way Forward
    - Create action plan
    - Establish timeline
    - Define success metrics
    Key Questions:
    - "What will you do next?"
    - "When will you do it?"
    - "How will you overcome obstacles?"
    """,

    "Co-Active Coaching": """
    Core Principles:
    1. The Client is Naturally Creative, Resourceful, and Whole
    - Clients have the answers within themselves
    - Coach's role is to help uncover and leverage inner wisdom
    - Focus on the client's inherent capabilities

    2. Focus on the Whole Person
    - Address all aspects of life: work, relationships, personal growth
    - Recognize interconnectedness of different life areas
    - Balance being and doing

    3. Dance in This Moment
    - Stay present and responsive
    - Work with what emerges
    - Embrace uncertainty and possibility

    Key Components:
    1. Fulfillment Coaching
    - Helping clients identify and live their values
    - Exploring life purpose and meaning
    - Creating alignment between actions and values

    2. Balance Coaching
    - Exploring different perspectives
    - Moving from current to desired state
    - Making conscious choices

    3. Process Coaching
    - Working with emotions and energy
    - Staying present with difficulty
    - Transforming limiting patterns
    """,

    "Emotional Intelligence Framework": """
    Based on Daniel Goleman's Framework:

    1. Self-Awareness
    - Emotional self-awareness
    - Accurate self-assessment
    - Self-confidence
    Development techniques:
    - Mindfulness practices
    - Reflection exercises
    - Feedback analysis

    2. Self-Management
    - Emotional self-control
    - Adaptability
    - Achievement orientation
    - Positive outlook
    Development strategies:
    - Trigger identification
    - Response planning
    - Stress management techniques

    3. Social Awareness
    - Empathy
    - Organizational awareness
    - Service orientation
    Enhancement methods:
    - Active listening
    - Body language reading
    - Context observation

    4. Relationship Management
    - Influence
    - Conflict management
    - Teamwork
    Development approaches:
    - Communication skills
    - Networking
    - Collaboration techniques
    """,

    "Solution-Focused Techniques": """
    Core Principles:
    1. Focus on Solutions Rather Than Problems
    - Identify what works
    - Build on past successes
    - Create forward momentum

    2. Future Orientation
    - Envision preferred future
    - Set clear objectives
    - Create action steps

    3. Resource Activation
    - Identify existing strengths
    - Leverage past successes
    - Build on what works

    Key Techniques:
    1. The Miracle Question
    "Suppose tonight while you sleep, a miracle happens. When you wake up tomorrow, what would be different that would tell you life had suddenly gotten better?"

    2. Scaling Questions
    - Use 1-10 scales to measure progress
    - Identify small steps forward
    - Build confidence through recognition of progress

    3. Exception Finding
    - Identify times when the problem is less severe
    - Analyze what works in those situations
    - Replicate successful strategies
    """,

    "Neuroscience-Based Coaching": """
    Brain-Based Principles:
    1. Neuroplasticity
    - Understanding brain's ability to change
    - Creating new neural pathways
    - Importance of repetition and practice

    2. Threat and Reward Response
    - SCARF Model (Status, Certainty, Autonomy, Relatedness, Fairness)
    - Managing emotional triggers
    - Creating psychological safety

    3. Attention and Focus
    - Role of focused attention in change
    - Impact of stress on decision-making
    - Importance of reflection and integration

    Application Techniques:
    1. State Management
    - Understanding brain states
    - Techniques for optimal performance
    - Stress reduction methods

    2. Goal Pursuit
    - Brain-friendly goal setting
    - Habit formation
    - Progress monitoring

    3. Learning Integration
    - Reflection practices
    - Consolidation techniques
    - Application planning
    """
}


@st.cache_resource
def initialize_vector_store():
    """
    Initialize the vector store with coaching methodologies.
    Uses Streamlit caching to persist across sessions.
    """
    # Create documents from the methodologies
    documents = [
        Document(page_content=content, metadata={"source": title})
        for title, content in COACHING_METHODOLOGIES.items()
    ]
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store


def get_chain_response(question: str, retriever, chain, skill_area: str, current_level: str, specific_goal: str):
    """
    Get response from the chain using retriever and input context.
    """
    # Get relevant documents
    docs = retriever.get_relevant_documents(question)
    context = "\n".join(doc.page_content for doc in docs)
    
    # Format input with context and user parameters
    formatted_input = f"""
    Context from coaching resources:
    {context}
    
    Skill Area: {skill_area}
    Current Level: {current_level}
    Specific Goal: {specific_goal}
        
    User Question: {question}
    """
    
    # Run chain with formatted input
    response = chain.run(input=formatted_input)
    return {"answer": response, "source_documents": docs}

@st.cache_resource
def setup_chat_rag_chain(_vector_store: FAISS):
    """
    Alternative RAG chain setup using ChatPromptTemplate.
    Uses Streamlit caching to persist the chain.
    """
    llm = ChatOpenAI(temperature=0.5, model_name="gpt-4", streaming=True)
    
    # Create retriever
    retriever = _vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Initialize conversation memory
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=msgs
    )
    
    # Create chain with existing coaching template
    chain = LLMChain(
        llm=llm,
        prompt=coaching_template,
        memory=memory,
        verbose=True
    )
    
    return chain, retriever

def main():
    # Set streamlit page configuration
    st.set_page_config(page_title="Human Skills Coach")

    # Add form inputs before creating the prompt and chain
    with st.sidebar:
        st.markdown("## Set Your Coaching Context")
        skill_area = st.selectbox(
            "What skill area would you like to work on?",
            ["Public Speaking", "Influencing", "Communication", "Self Reflection", "Active Listening", "Creative Thinking"]
        )
        current_level = st.select_slider(
            "Current skill level",
            options=["Beginner", "Intermediate", "Advanced"]
        )
        specific_goal = st.text_area("What specific goal would you like to achieve?")

    try:
        # Initialize vector store using Streamlit caching
        vector_store = initialize_vector_store()
        
        # Setup RAG chain with caching
        chain, retriever = setup_chat_rag_chain(vector_store)
        
        # Streamlit UI
        left_co, cent_co, last_co = st.columns(3)
        with cent_co:
            st.image("eque.png")

        st.title("""Your personal *human skills* coach.""")
        st.markdown("""As AI handles the technical day-to-day, human skills matter more than ever. 
                    Practice your interpersonal skills with Eque to nail your next interview, lead confidently, 
                    or communicate more effectively. No pressure – just guided practice via text (voice coming soon).
                    Build the interpersonal skills that set you apart.
            """
            , unsafe_allow_html=False, help=None)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message.get("avatar")):
                st.markdown(message["content"])

        # Chat input and response
        if prompt := st.chat_input("Ask me what I can help you with!"):
            # Add user message to chat history
            st.session_state.messages.append({
                "role": "human",
                "content": prompt,
                "avatar": "🧑‍💻"
            })
            
            # Display user message
            with st.chat_message("human", avatar="🧑‍💻"):
                st.markdown(prompt)

            # Get response using RAG chain
            with st.spinner("Thinking..."):
                response = get_chain_response(prompt, retriever, chain, skill_area, current_level, specific_goal)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "avatar": "🤖"
                })
                
            # Display assistant response
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response["answer"])
                    
                # Show sources in an expander
                #with st.expander("View sources"):
                    #for doc in response["source_documents"]:
                        #st.markdown(f"**Source:** {doc.metadata['source']}")
                        #st.markdown(doc.page_content)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    # Add credit
    with st.sidebar:
        st.markdown('''Made with 💖 by Ash Stevens  
        https://github.com/ashjstevens/human-skills
        ''')

if __name__ == "__main__":
    main()