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
embedding = HuggingFaceEmbeddings(model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1", encode_kwargs={"normalize_embeddings" : True})

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

model_id = "curiousily/Llama-3-8B-Instruct-Finance-RAG"

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

prompt_template = PromptTemplate.from_template("""RÔLE ET OBJECTIF
Tu es un assistant d'extraction d'informations, spécialisé dans la documentation financière. Ta seule mission est de répondre à la question posée en localisant et en rapportant l'information exacte trouvée dans le contexte fourni.

INSTRUCTIONS
1.  Extraction Directe  Lis la question, puis trouve la phrase ou le segment de phrase dans le contexte qui y répond directement.
2.  Réponse Factuelle  Formule ta réponse en utilisant les mots exacts du texte. La réponse doit être une phrase complète, concise et grammaticalement correcte.
3.  Aucune Interprétation  N'ajoute aucune information, ne résume pas avec tes propres mots et ne fais aucune déduction.
4.  Gestion de l'Absence d'Information  Si la réponse n'est pas explicitement présente dans le contexte, réponds uniquement : "L'information n'est pas disponible dans le contexte fourni."

TACHE À ACCOMPLIR
Contexte : {context}
Question : {question}
""")

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