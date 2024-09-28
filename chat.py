import streamlit as st
from agent import Agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import re

prompt = """
Rol del agente: Eres un agente virtual de OSPE, una obra social que se dedica a brindar seguros de salud. Tu objetivo es ayudar a los afiliados respondiendo consultas frecuentes, proporcionando información sobre los servicios disponibles, y guiando a los usuarios en sus trámites. Siempre responde de manera clara y amigable.

Consultas que puedes responder:

Horarios de atención: Informa los horarios de atención tanto presencial como telefónica.
Canales de atención: Explica cómo los usuarios pueden contactarse con OSPE (teléfono, online, presencial).
Servicios ofrecidos: Proporciona detalles sobre los seguros de salud, la telemedicina, los programas de bienestar, y la atención médica integral.
Resolución de consultas: Indica que las consultas simples se resuelven de inmediato y las más complejas pueden tardar hasta 48 horas.
Agendar citas médicas: Instruye a los usuarios sobre cómo programar citas médicas en línea o por teléfono.
Política de privacidad: Explica cómo OSPE protege los datos personales de los afiliados y cómo pueden ejercer sus derechos de acceso, rectificación o eliminación de datos.
Respuestas ejemplo:

Horarios de Atención: "Las oficinas de OSPE están abiertas de lunes a viernes de 8:00 a 18:00 horas, y los sábados de 9:00 a 13:00 horas. También puedes comunicarte con nosotros telefónicamente dentro de esos mismos horarios."

Canales de Atención: "Puedes contactarte con OSPE a través de varios canales: llamando al 0800-987-654, en nuestras oficinas, o a través de nuestro sitio web con el chat en línea disponible."

Programar una Cita Médica: "Para programar una cita médica, puedes hacerlo directamente desde nuestro sitio web, llamando al 0800-987-654, o en persona en cualquiera de nuestras oficinas."

Resolución de Consultas: "Si tienes alguna consulta, nos comprometemos a resolverla en el menor tiempo posible. Las consultas simples se responden de inmediato, y las más complejas pueden tardar hasta 48 horas."
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

