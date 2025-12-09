
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
from langchain.tools import Tool
from langchain import hub
from langchain.prompts import PromptTemplate
import sentence_transformers

from huggingface_hub import login
from rag_engine.llm_loader import load_llm
from rag_engine.retriever import rerank

from transformers import StoppingCriteria, StoppingCriteriaList

"""
from transformers import pipeline
from transformers import BitsAndBytesConfig
from pinecone.exceptions import NotFoundException
from rag_engine.embeddings import embed_texts
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

def smart_retrieval(query: str) -> str:
    # 1. Recherche large dans Pinecone (k=20 au lieu de 5)
    docs = vector_store.similarity_search(query, k=20)
    
    # 2. Utilisation de votre fonction rerank existante pour garder le Top 5 pertinent
    # (C'est ça qui restaure la pertinence perdue)
    sorted_docs = rerank(query, docs, use_rerank=True)[:5]
    
    # 3. On formate le texte proprement pour l'LLM
    context = ""
    for doc in sorted_docs:
        source = doc.metadata.get("source", "Inconnu")
        context += f"Source: {source}\nContenu: {doc.page_content}\n\n"
    return context

# Création de l'outil personnalisé
tool_smart_retriever = Tool(
    name="recherche_expert_finance",
    func=smart_retrieval,
    description="Utilise cet outil pour obtenir directement les informations contenues dans les documents. Ne nécessite pas d'étape d'extraction supplémentaire."
)

tools = [tool_smart_retriever]

# 5 chargement du LLM
# ✅ Nouveau chargement optimisé
TOK, LLM_MODEL = load_llm()

from transformers import pipeline, StoppingCriteria, StoppingCriteriaList

#creation de la classe critères d'arrets
class StopOnString(StoppingCriteria):
    """
    Critère d'arrêt robuste qui vérifie une large fenêtre de texte 
    pour empêcher le modèle de parler à la place de l'outil.
    """
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores, **kwargs):
        # On décode les 100 derniers tokens pour être sûr de capter le motif
        # même si le modèle le génère morceau par morceau
        window_size = 100
        start_idx = max(0, input_ids.shape[1] - window_size)
        generated_text = self.tokenizer.decode(input_ids[0][start_idx:], skip_special_tokens=True)
        
        for s in self.stop_strings:
            if s in generated_text:
                return True
        return False

# Initialisation des critères d'arrêt
# On veut qu'il s'arrête dès qu'il tente d'écrire "Observation:" lui-même

stop_words = ["Observation:", "\nObservation:", "Observation", "Final Answer:", "\nFinal Answer:"]
stopping_criteria = StoppingCriteriaList([
    StopOnString(stop_words, TOK)
])

pipe = pipeline(
    "text-generation",
    model=LLM_MODEL,
    tokenizer=TOK,
    max_new_tokens=512,
    do_sample=False,
    return_full_text=False,
    pad_token_id=TOK.eos_token_id,
    stopping_criteria=stopping_criteria
)

llm = HuggingFacePipeline(
    pipeline=pipe, 
    model_kwargs={"stop": stop_words}
)


#6 création de l'agent

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question.

IMPORTANT GUIDELINES:
1. Provide a short and precise answer based SOLELY on the "Observation" from the tool.
2. If the tool returns "Unknown" or low relevance data, say "Unknown".
3. Do not make up information.
4. STOP generating immediately after writing "Action Input". DO NOT write "Observation" yourself.

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# La création de l'agent reste identique, mais elle utilisera ce nouveau prompt
agent = create_react_agent(llm, tools, prompt)


agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5# Important pour les modèles locaux qui formatent parfois mal la sortie
)

def rag_pipeline(query):
    try:
        response = agent_executor.invoke({"input": query})
        return response["output"]
    except Exception as e:
        return f"Erreur lors de l'exécution de l'agent : {str(e)}"

