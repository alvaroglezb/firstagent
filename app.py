import streamlit as st
from dotenv import load_dotenv
import base64
load_dotenv()
import asyncio
import os
from datetime import datetime
from agents import InputGuardrail, GuardrailFunctionOutput, Agent, Runner, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled
from agents.exceptions import InputGuardrailTripwireTriggered
from openai import AsyncAzureOpenAI
import requests
from pydantic import BaseModel
from typing import Literal

# Configuración de la página
st.set_page_config(
    page_title="Chatbot Álvaro - OpenAI Agents",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:alvaroglezb@gmail.com'
    }
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .user-message {
    background-color: #D44E4E;   /* gris azulado muy claro */
    border-left-color: #F5F5F5; /* azul grisáceo */
    }

    .assistant-message {
    background-color: #4E81D4;   /* gris lila claro */
    border-left-color: #F5F5F5;  /* púrpura apagado */
    }
    
    .error-message {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    
    .warning-message {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    
    .success-message {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    
    .metrics-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

## --- CONFIGURACIÓN INICIAL ---##
# Context sobre Álvaro
CONTEXT = """
# Perfil Personal - Álvaro González Bielza

## 📋 Información Básica
| Campo | Información |
|-------|-------------|
| **Nombre** | Álvaro |
| **Primer Apellido** | González |
| **Segundo Apellido** | Bielza |
| **Nombre Completo** | Álvaro González Bielza |

## 📏 Características Físicas
- **Altura**: 1,75 cm
- **Peso**: 70 Kg

## 🎨 Preferencias
- **Color Favorito**: Azul
Aquí tienes el contenido del CV convertido a formato Markdown:

---

# Álvaro González Bielza

**Ingeniero Industrial**
📞 651191316 | 📧 [alvaroglezb@gmail.com](mailto:alvaroglezb@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/alvarogonzalezbielza)

---

## Perfil

Consultor tecnológico con sólida experiencia en automatización inteligente e inteligencia artificial generativa. Especializado en todo el ciclo de vida de un proyecto de automatización con herramientas como UiPath y Azure. Experiencia en gestión de equipos multidisciplinarios e implementación de múltiples soluciones en entornos productivos en cliente final.

---

## Experiencia

### CONSULTOR SENIOR DTO. DIGITAL ENGINEERING | EY (Ernst & Young)

**Diciembre de 2021 – Presente**

**Proyecto destacado:** Automatización Inteligente en la Agencia Digital de Andalucía

* **Rol:** Responsable de negocio (habiendo pasado por desarrollador y consultor). Equipo formado por 40 personas, responsable de 16 consultores y desarrolladores.
* **Cliente:** Agencia Digital de Andalucía
* **Periodo:** 2021 – presente
* **Logros clave:**

  * Ahorro generado > 40M €
  * Automatización de > 100 procesos
  * Procesamiento de > 3M de documentos
  * Iniciativa UAI DOC: Tratamiento documental con IA (Premio Socinfo)
  * Impacto eficiencia: > 2,4 M expedientes procesados
  * Desarrollo de componentes reutilizables entre consejerías para facilitar escalabilidad
* **Retos:**

  * Alto volumen de procesos en planificación
  * Inclusión de cuadros de mando con Kibana
  * Cliente presente en múltiples consejerías
  * Puesta en producción de solución con IA generativa
  * Balancear la carga del equipo según desarrollos en curso

---

## Educación

* **Máster Habilitante en Ingeniería Industrial**
  *2020 – 2022 | Universidad Politécnica de Madrid*

* **Grado en Ingeniería en Organización Industrial**
  *2015 – 2019 | Universidad de Sevilla*

---

## Certificaciones y Cursos Adicionales

* **Microsoft AI-102 AI Engineer Associate**
  ID: B37808685713A12D

* **Founderz Master Online IA e Innovación**
  ID: e85010b5-a094-40d0-9a92-1e391635c242

* **Formador en curso de RPA para la Junta de Extremadura (150h)**

* **IBM Watsonx Orchestrate Sales Foundation**

* **McKinsey Forward Program**

* **EY Artificial Intelligence – AI Engineering**

* **UiPath RPA Developer Foundation**

* **Professional Scrum Master** *(en preparación)*

---

## Aptitudes y Habilidades

### Hard Skills

* **Azure AI Services (Avanzado):**
  Document Intelligence, AI Search, Azure OpenAI, Key Vault, Azure Storage, Azure Functions
* **UiPath (Avanzado):**
  ReFramework, Orchestrator
* **Python (Intermedio):**
  Funciones Azure, informes con APIs y Pandas
* **SQL (Avanzado):**
  Queries para integración con RPA y Azure, esquemas E/R (3FN)
* **Consumo de APIs (Avanzado)**
* **Power Platform (Básico)**
* **Kibana (Básico)**
* **Documentación de ciclo de vida de proyectos:**
  PDD, SDD, UAT, PRN
* **Azure DevOps (Avanzado):**
  Gestión de proyectos, consumo por APIs, informes

### Soft Skills

* **Inglés (Avanzado)**
* Comunicación efectiva con cliente, comprensión de necesidades
* Reporting semanal de KPIs y estado de proyecto
* Capacidad de explicar ideas técnicas a perfiles diversos
* Propuestas de mejora y evaluación de procesos
* Gestión de equipos multidisciplinares
* Resolución de incidencias con análisis y calma
* Priorización y gestión simultánea de múltiples proyectos
* **Aprendizaje continuo:** Podcast, libros, certificaciones, etc.

---

¿Te gustaría que lo formatee como un archivo `.md` descargable o subirlo a alguna plataforma?

"""

# Cliente Azure OpenAI
@st.cache_resource
def get_openai_client():
    return AsyncAzureOpenAI(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        api_version="2024-12-01-preview"
    )

## --- MODELOS PYDANTIC ---##
class WeatherResponse(BaseModel):
    Ciudad: str
    Temperatura: float

class BookAuthor(BaseModel):
    Autor: str
    Titulo: str

class ContentModeration(BaseModel):
    is_appropiate: bool
    reasoning: str
    risk_level: Literal['low', 'medium', 'high']

## --- FUNCIONES HERRAMIENTAS ---##
def get_image_as_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_base64 = get_image_as_base64("1700566366811.jpg")
image_html = f'<img src="data:image/png;base64,{image_base64}" width="24" style="vertical-align:middle; margin-right:8px;">'
@function_tool
def get_weather(latitude, longitude) -> str:
    try:
        response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m",verify=False)
        data = response.json()
        return data['current']['temperature_2m']
    except Exception as e:
        error_msg= f"Error retrieving weather data: {e}"
        return error_msg

@function_tool
def get_book_author(book_title: str) -> str:
    """Obtiene el autor de un libro dado su título"""
    try:
        response = requests.get(f"https://openlibrary.org/search.json?q={book_title}", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('docs') and len(data['docs']) > 0:
            return str(data['docs'][0].get('author_name', ['Autor desconocido'])[0])
        return "Autor no encontrado"
    except Exception as e:
        return f"Error buscando el autor: {str(e)}"

def assemble_conversation(result,query):
    if result != None:
        query = result.to_input_list() + [{'role':"user",'content':query}]
    else:
        query = [{'role':"user",'content':query}]
    return query


## --- GUARDRAILS ---##
async def content_moderation_function(ctx, agent, query):
    """Función de moderación de contenido"""
    try:
        content_to_moderate = ""
        if isinstance(query, list):
            for item in query:
                if isinstance(item, dict) and item.get('role') == 'user':
                    content_to_moderate = item.get('content', '')
                    break
        elif isinstance(query, str):
            content_to_moderate = query
        else:
            content_to_moderate = str(query)
        
        result = await Runner.run(st.session_state.content_moderation_agent, content_to_moderate, context=ctx.context)
        final_output = result.final_output_as(ContentModeration)
        
        return GuardrailFunctionOutput(
            output_info=final_output,
            tripwire_triggered=not final_output.is_appropiate,
        )
    except Exception as e:
        return GuardrailFunctionOutput(
            output_info=ContentModeration(
                is_appropiate=True,
                reasoning=f"Error en moderación: {str(e)}",
                risk_level="low"
            ),
            tripwire_triggered=False,
        )

## --- INICIALIZACIÓN DE AGENTES ---##
@st.cache_resource
def initialize_agents():
    """Inicializa todos los agentes"""
    client = get_openai_client()
    set_tracing_disabled(disabled=True)
    
    # Agente meteorológico
    Weather_Agent= Agent(
        name="Agente Meteorologico",
        instructions="Eres un agente encargado de devolver la temperatura de la ciudad solicitada",
        model=OpenAIChatCompletionsModel(
            model="GPT-4o-structured",
            openai_client=client),
        tools=[get_weather],
        output_type=WeatherResponse
    )

    # Agente bibliotecario
    library_agent = Agent(
        name="Agente Bibliotecario",
        instructions="Respondes con el autor del libro solicitado.",
        model=OpenAIChatCompletionsModel(model="GPT-4o-structured", openai_client=client),
        tools=[get_book_author],
        output_type=BookAuthor
    )
    
    # Agente de Álvaro
    alvaro_agent = Agent(
        name="Agente de Álvaro",
        instructions=f"Eres un agente que responde preguntas sobre Álvaro basándote en la información que tienes de él.\n\nInformación:{CONTEXT}",
        model=OpenAIChatCompletionsModel(model="GPT-4o-structured", openai_client=client),
    )
    
    # Agente de moderación
    content_moderation_agent = Agent(
        name="Content Moderation Agent",
        instructions="""Eres un agente encargado de moderar el contenido de las conversaciones.
        Evalúa si el contenido es apropiado y clasifica el riesgo como low, medium o high.
        Si el contenido es inapropiado, establece is_appropiate=False.""",
        model=OpenAIChatCompletionsModel(model="GPT-4o-structured", openai_client=client),
        output_type=ContentModeration,
    )
    
    # Agente principal
    main_agent = Agent(
        name="Álvaro Agent",
        instructions=(
            "Eres un agente que utiliza las diferentes funciones para obtener la información solicitada. "
            "Utiliza las herramientas apropiadas según el tipo de consulta."
        ),
        model=OpenAIChatCompletionsModel(model="GPT-4o-structured", openai_client=client),
        tools=[
            Weather_Agent.as_tool(
                tool_name="Agente_Meteorologico",
                tool_description="Agente para consultas meteorológicas"
            ),
            alvaro_agent.as_tool(
                tool_name="Agente_de_Alvaro",
                tool_description="Agente para preguntas sobre Álvaro"
            ),
            library_agent.as_tool(
                tool_name="Agente_Bibliotecario",
                tool_description="Agente para consultas sobre libros"
            ),
        ],
        input_guardrails=[InputGuardrail(guardrail_function=content_moderation_function)]
    )
    
    return {
        "main_agent": main_agent,
        "content_moderation_agent": content_moderation_agent,
        "weather_agent": Weather_Agent,
        "library_agent": library_agent,
        "alvaro_agent": alvaro_agent
    }

## --- FUNCIONES DE LA INTERFAZ ---##
def initialize_session_state():
    """Inicializa el estado de la sesión"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_result" not in st.session_state:
        st.session_state.conversation_result = None
    
    if "interaction_count" not in st.session_state:
        st.session_state.interaction_count = 0
    
    if "agents_initialized" not in st.session_state:
        agents = initialize_agents()
        for key, agent in agents.items():
            setattr(st.session_state, key, agent)
        st.session_state.agents_initialized = True

def display_message(role, content, timestamp=None, message_type="normal"):
    """Muestra un mensaje en el chat"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M:%S")
    
    if role == "user":
        icon = "👤"
        css_class = "user-message"
    else:
        icon = image_html
        css_class = "assistant-message"
    
    if message_type == "error":
        css_class = "error-message"
        icon = "❌"
    elif message_type == "warning":
        css_class = "warning-message"
        icon = "⚠️"
    elif message_type == "success":
        css_class = "success-message"
        icon = "✅"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{icon} {role.title()}</strong> <small>({timestamp})</small><br>
        {content}
    </div>
    """, unsafe_allow_html=True)

async def process_message(user_input):
    """Procesa un mensaje del usuario"""
    try:
        # Ensamblar conversación
        formatted_query = assemble_conversation(st.session_state.conversation_result, user_input)
        
        # Ejecutar agente principal
        result = await Runner.run(st.session_state.main_agent, formatted_query)
        
        # Actualizar estado
        st.session_state.conversation_result = result
        st.session_state.interaction_count += 1
        
        return result.final_output, None
        
    except InputGuardrailTripwireTriggered as e:
        st.session_state.interaction_count += 1
        return f"\n🛡️ CONTENIDO BLOQUEADO POR SEGURIDAD {e.guardrail_result.output.output_info.reasoning}", None
    except Exception as e:
        return None, f"Error inesperado: {str(e)}"

## --- INTERFAZ PRINCIPAL ---##
def main():
    """Función principal de la interfaz"""
    initialize_session_state()
    
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Chatbot Álvaro - OpenAI Agents SDK</h1>
        <p>Asistente inteligente con protecciones de seguridad integradas</p>
        <p>PROPIEDAD INTELECTUAL DE ALVARO GONZALEZ BIELZA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar con información
    with st.sidebar:
        st.header("📊 Información del Sistema")
        
        # Métricas
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Interacciones", st.session_state.interaction_count)
        with col2:
            st.metric("Mensajes", len(st.session_state.messages))
        
        st.markdown("---")
        
        # Capacidades del bot
        st.header("🎯 Capacidades")
        st.markdown("""
        - **📊 Información sobre Álvaro González Bielza**
        - **🌤️ Consultas meteorológicas**
        - **📚 Información sobre libros y autores**
        - **🛡️ Moderación de contenido automática**
        """)
        
        st.markdown("---")
        
        # Controles
        st.header("⚙️ Controles")
        if st.button("🗑️ Limpiar Conversación", type="secondary"):
            st.session_state.messages = []
            st.session_state.conversation_result = None
            st.session_state.interaction_count = 0
            st.rerun()
        
        # Información técnica
        st.header("🔧 Información Técnica")
        st.info(f"""
        **Modelo**: GPT-4o-structured  
        **Guardrails**: Activos  
        **Estado**: {'🟢 Conectado' if st.session_state.agents_initialized else '🔴 Desconectado'}
        """)
    
    # Área principal de chat
    st.header("💬 Conversación")
    
    # Contenedor de mensajes
    chat_container = st.container()
    
    with chat_container:
        # Mostrar mensajes existentes
        for message in st.session_state.messages:
            display_message(
                message["role"], 
                message["content"], 
                message.get("timestamp"),
                message.get("type", "normal")
            )
    
    # Input del usuario
    st.markdown("---")
    
    # Formulario para enviar mensajes
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Escribe tu mensaje:",
                placeholder="Pregúntame sobre Álvaro, el clima o libros...",
                key="user_input"
            )
        
        with col2:
            submit_button = st.form_submit_button("Enviar 📤", type="primary")
    
    # Procesar mensaje
    if submit_button and user_input:
        # Agregar mensaje del usuario
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp,
            "type": "normal"
        })
        
        # Mostrar indicador de procesamiento
        with st.spinner("🔍 Procesando con guardrails de seguridad..."):
            # Procesar mensaje de forma asíncrona
            response, error = asyncio.run(process_message(user_input))
        
        if error:
            # Mostrar error
            st.session_state.messages.append({
                "role": "sistema",
                "content": error,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "type": "error"
            })
        else:
            # Mostrar respuesta
            st.session_state.messages.append({
                "role": "asistente",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "type": "normal"
            })
        
        # Recargar la página para mostrar los nuevos mensajes
        st.rerun()
    
    # Ejemplos de consultas
    if len(st.session_state.messages) == 0:
        st.header("💡 Ejemplos de consultas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("¿Cuál es el color favorito de Álvaro?"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": "¿Cuál es el color favorito de Álvaro?",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "type": "normal"
                })

                # Mostrar indicador de procesamiento
                with st.spinner("🔍 Procesando con guardrails de seguridad..."):
                # Procesar mensaje de forma asíncrona
                    response, error = asyncio.run(process_message("¿Cuál es el color favorito de Álvaro?"))

                # Mostrar respuesta
                st.session_state.messages.append({
                "role": "asistente",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "type": "normal"
                })
                st.rerun()
        
        with col2:
            if st.button("¿Qué temperatura hace en Madrid?"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": "¿Qué temperatura hace en Madrid? (latitud: 40.4168, longitud: -3.7038)",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "type": "normal"
                })
                # Mostrar indicador de procesamiento
                with st.spinner("🔍 Procesando con guardrails de seguridad..."):
                # Procesar mensaje de forma asíncrona
                    response, error = asyncio.run(process_message("¿Qué temperatura hace en Madrid?"))

                # Mostrar respuesta
                st.session_state.messages.append({
                "role": "asistente",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "type": "normal"
                })
                st.rerun()
        
        with col3:
            if st.button("¿Quién escribió El Quijote?"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": "¿Quién escribió Don Quijote de la Mancha?",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "type": "normal"
                })
                # Mostrar indicador de procesamiento
                with st.spinner("🔍 Procesando con guardrails de seguridad..."):
                # Procesar mensaje de forma asíncrona
                    response, error = asyncio.run(process_message("¿Quién escribió Don Quijote de la Mancha?"))

                # Mostrar respuesta
                st.session_state.messages.append({
                "role": "asistente",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "type": "normal"
                })
                st.rerun()

if __name__ == "__main__":
    main()