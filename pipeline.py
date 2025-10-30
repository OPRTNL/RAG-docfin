import transformers
import torch
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline
from transformers import BitsAndBytesConfig
import os
from langchain_pinecone.vectorstores import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login # <-- Import the login function
from rag_engine.llm_loader import load_llm
from rag_engine.embeddings import embed_texts
from rag_engine.retriever import rerank

# Charger le modèle d'encodage de texte BAAI/bge-small-en-v1.5 de HuggingFace
embedding = HuggingFaceEmbeddings(model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
                                  model_kwargs={'device': 'cuda'}, # Pin the model to the GP
                                  encode_kwargs={
                                      'normalize_embeddings': True,
                                      'batch_size': 32 # Process queries in batches
                                  })

load_dotenv()
pinecone_api_key = os.getenv('PINECONE_API_KEY')
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')

login(token=huggingface_api_key)

# Initialiser le VectorStore de LlamaIndex avec l'index de Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)
pinecone_index = pinecone.Index("rag")
vector_store = PineconeVectorStore(
    index=pinecone_index,
    embedding = embedding
)

# ✅ Nouveau chargement optimisé
TOK, LLM = load_llm()

from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline

llm = HuggingFacePipeline(
    pipeline=pipeline(
        "text-generation",
        model=LLM,
        tokenizer=TOK,
        max_new_tokens=128,
        do_sample=False,
        return_full_text=False
    )
)


from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("""Provide a short and precise answer to the following question, based solely on the information from the documents below:

---------------------
{context}
---------------------

Use the information in these documents to answer the question in a factual and concise manner. If the answer to the question is not contained within these documents, respond simply with "Unknown.

Question: {question}""")

def build_context(docs):
    context = ""
    for doc in docs:
        context += "titre du document :" + doc.metadata["source"]
        context += "\n"
        context += doc.page_content
        context += "\n\n"
    return context
        
def rag_pipeline(query):
    # Recherche Pinecone
    docs = vector_store.similarity_search(query, k=20) # large pool
    # rerank désactivé par défaut pour vitesse
    retrieved_docs = rerank(query, docs, use_rerank=True)[:5]
    
    # Construit le contexte
    context = build_context(retrieved_docs)

    # Template
    prompt = prompt_template.invoke({
        "question": query,
        "context": context
    })

    # Inference
    return llm.invoke(prompt).strip()
