
import os
from dotenv import load_dotenv
from pinecone import Pinecone

#import transformers
#import torch
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain import hub

from huggingface_hub import login
from rag_engine.llm_loader import load_llm

"""
from transformers import pipeline
from transformers import BitsAndBytesConfig
from pinecone.exceptions import NotFoundException
from rag_engine.embeddings import embed_texts
from rag_engine.retriever import rerank
"""

# 1. Configuration et Login
load_dotenv()
pinecone_api_key = os.getenv('PINECONE_API_KEY')
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')

login(token=huggingface_api_key)

# 2. Initialiser les Embeddings
# Charger le modèle d'encodage de texte BAAI/bge-small-en-v1.5 de HuggingFace
embedding = HuggingFaceEmbeddings(model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
                                  model_kwargs={'device': 'cuda'}, # Pin the model to the GP
                                  encode_kwargs={
                                      'normalize_embeddings': True,
                                      'batch_size': 32 # Process queries in batches
                                  })




# 3. Initialiser Pinecone et VectorStore
pinecone = Pinecone(api_key=pinecone_api_key)
pinecone_index = pinecone.Index("rag")
vector_store = PineconeVectorStore(
    index=pinecone_index,
    embedding = embedding
)

# 4 outils retriver pour l'agent
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

tool_retriever = create_retriever_tool(
    retriever,
    "recherche_documents_financiers",
    "Utilise cet outil pour rechercher des informations dans les documents financiers indexés. Pose une question précise."
)

tools = [tool_retriever]

# 5 chargement du LLM
# ✅ Nouveau chargement optimisé
TOK, LLM_MODEL = load_llm()

from transformers import pipeline


pipe = pipeline(
    "text-generation",
    model=LLM_MODEL,
    tokenizer=TOK,
    max_new_tokens=512,
    do_sample=False,
    return_full_text=False,
    pad_token_id=TOK.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)


#6 création de l'agent
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True # Important pour les modèles locaux qui formatent parfois mal la sortie
)

def rag_pipeline(query):
    try:
        response = agent_executor.invoke({"input": query})
        return response["output"]
    except Exception as e:
        return f"Erreur lors de l'exécution de l'agent : {str(e)}"

