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

# Charger le modèle d'encodage de texte BAAI/bge-small-en-v1.5 de HuggingFace
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings" : True})

load_dotenv()
pinecone_api_key = os.getenv('PINECONE_API_KEY')
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')


# Initialiser le VectorStore de LlamaIndex avec l'index de Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)
pinecone_index = pinecone.Index("rag")
vector_store = PineconeVectorStore(
    index=pinecone_index,
    embedding = embedding
)

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Chargement de la configuration du modèle
model_config = transformers.AutoConfig.from_pretrained(model_id)

# Initialiser le tokeniseur
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    device_map='auto'
)

llm=HuggingFacePipeline(
    pipeline=pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        do_sample=False,
        return_full_text=False  # Très important ! On ne veut pas le prompt initial
    )
)

from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer: """)

def build_context(docs):
    context = ""
    for doc in docs:
        context += "titre du document :" + doc.metadata["source"]
        context += "\n"
        context += doc.page_content
        context += "\n\n"
        
def rag_pipeline(query):
    # Tout d'abord, on recherche les documents
    retrieved_docs = vector_store.similarity_search(query)
    # Ensuite, on injecte les documents dans le prompt
    prompt = prompt_template.invoke({
        "question": query,
        "context": build_context(retrieved_docs)
    })
    # Enfin, on envoit l'intégralité du prompt au LLM
    return llm.invoke(prompt).strip()

query = """
Quel produit a l'identifiant LU1437017350 ? Donne également le titre du document utilisé pour ta réponse.
"""

# Effectuer une requête
response = rag_pipeline(query)
print(response)