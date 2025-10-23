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

model_id = "CohereForAI/c4ai-command-r-v01"

# Chargement de la configuration du modèle
model_config = transformers.AutoConfig.from_pretrained(model_id)

# Initialiser le tokeniseur
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# Load the model with quantization
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=quantization_config, # Apply the config here
    device_map='auto'
)

llm=HuggingFacePipeline(
    pipeline=pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        do_sample=False,
        return_full_text=False  # Très important ! On ne veut pas le prompt initial
    )
)

from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("""
Tu es un assistant d'extraction d'informations, spécialisé dans la documentation financière. Ta seule mission est de répondre en à la question en localisant et en rapportant l'information exacte trouvée dans le context fourni. Formule ta réponse en utilisant les mots exacts du texte. La réponse doit être une phrase complète, concise et grammaticalement correcte.

Contexte : {context}
Question : {question}
Réponse :
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
