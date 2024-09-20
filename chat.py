import streamlit as st
from agent import Agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import re

prompt = """
Eres un asistente de la nueva obra social llamada OSPE, tu trabajo es derivar a los pacientes que escriban por el chatbot consultando por turnos medicos, asistencia o por alguna consulta basica de medicina (por ejemplo que medicamento tomar si me duele la garganta). 
Deberias poder identificar que le sucede al paciente consultandole sus sintomas si es que los tiene y luego recomendarle un turno con la especialidad detectada.

"""

tools = None

if tools:
    agent = Agent(model_type="groq", prompt=prompt, tools=tools)
else:
    agent = Agent(model_type="groq", prompt=prompt)


st.title("Agente OSPE")




st.write("👋 ¡Hola! Bienvenido al chat de atención de OSPE. ¿En qué puedo ayudarte hoy?")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display all previous chat messages
for message in st.session_state.messages:
    if isinstance(message, (HumanMessage, AIMessage)) and message.content:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.markdown(message.content)


# React to user input
if prompt := st.chat_input("User input"):
    # Create a HumanMessage and add it to chat history
    human_message = HumanMessage(content=prompt)
    st.session_state.messages.append(human_message)

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Invoke the agent to get a list of AI messages, including potentially retrieved documents
    response_messages = agent.invoke(st.session_state.messages)

    # Update the session state with the new response
    st.session_state.messages = response_messages["messages"]

    # Display only the last AI message with content
    last_message = response_messages["messages"][-1]


    if isinstance(last_message, AIMessage) and last_message.content:
        st.chat_message("assistant").markdown(last_message.content)

