import logging
from token import STAR
from tracemalloc import start
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from decider import deciderRouter
from coneAgent import node_agent_pinecone
from apiAgent import node_agent_api
from generalAgent import node_agent_general
from settings import *

# Configuración básica de logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

class FriendlyResponder:
    def __init__(self):
        # Initialize the StateGraph and other components
        logging.info("Inicializando FriendlyResponder")
        try:
            self.state_graph = self.build_state_graph()
            logging.info("StateGraph construido con éxito")
        except Exception as e:
            logging.error(f"Error al construir StateGraph: {e}")

    def build_state_graph(self) -> StateGraph:
        try:
            logging.debug("Iniciando la construcción del StateGraph")
            graph_builder = StateGraph(State)
            graph_builder.add_node("coneAgentNode", node_agent_pinecone)
            graph_builder.add_node("apiAgentNode", node_agent_api)
            graph_builder.add_node("generalAgentNode", node_agent_general)
            graph_builder.add_conditional_edges(
                START,
                deciderRouter,
                {"Dataset": "coneAgentNode", "Api": "apiAgentNode", "General": "generalAgentNode"}
            )
            graph_builder.add_edge("coneAgentNode", END)
            graph_builder.add_edge("apiAgentNode", END)
            graph_builder.add_edge("generalAgentNode", END)
            logging.debug("StateGraph construido correctamente")
            return graph_builder.compile(checkpointer=memory)
        except Exception as e:
            logging.error(f"Error al construir el grafo: {e}")
            raise

    def respond(self, state: State) -> dict:
        try:
            question = state["messages"][-1].content
            logging.info(f"Recibida pregunta: {question}")
            response = settings.llm.invoke(question)
            logging.info(f"Respuesta generada: {response}")
            return {"messages": [response]}
        except Exception as e:
            logging.error(f"Error al procesar la respuesta: {e}")
            return {"messages": [f"Error: {e}"]}

    def run(self, message):
        session_id = message["session_id"]
        question = message["question"]
        config = {"configurable": {"thread_id": session_id}}

        logging.info(f"Iniciando run para session_id: {session_id} con la pregunta: {question}")

        responses = []
        try:
            events = self.state_graph.stream(
                {"messages": [("user", question)]},
                config,
                stream_mode="values"
            )
            for event in events:
                response_message = event["messages"][-1].content
                logging.debug(f"Evento procesado con mensaje: {response_message}")
                responses.append(response_message)
        except Exception as e:
            logging.error(f"Error durante el procesamiento del grafo de estado: {e}")
        
        if responses:
            logging.info(f"Última respuesta: {responses[-1]}")
            return responses[-1]
        else:
            logging.warning("No se generaron respuestas")
            return "No response"

# Example usage
if __name__ == "__main__":
    responder = FriendlyResponder()
    try:
        response = responder.run({"session_id": "12345", "question": "cuanto dolares son 22.000 cololmbianos"})
        logging.info(f"Respuesta final: {response}")
    except Exception as e:
        logging.error(f"Error en la ejecución principal: {e}")
