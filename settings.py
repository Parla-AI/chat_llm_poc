import os
from sys import flags 
from pinecone import Pinecone, ServerlessSpec
from langchain_mistralai.chat_models import ChatMistralAI
from pydantic import SecretStr
import time
# Obtener las credenciales de las variables de entorno
PINECONE_APIKEY = os.environ.get("PINECONE_APIKEY")
EXCHANGERATE_APIKEY = os.environ.get("EXCHANGERATE_APIKEY")
LLM_APIKEY = os.environ.get("LLM_APIKEY")



# Inicializar Pinecone con la API key
pc = Pinecone(api_key=PINECONE_APIKEY)


llm = ChatMistralAI(api_key=SecretStr(str(LLM_APIKEY)),model_name="mistral-large-latest")

