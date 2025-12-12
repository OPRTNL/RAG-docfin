
import os
import torch
from dotenv import load_dotenv
from pinecone import Pinecone

from langsmith import Client

from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.agents import create_agent #create_tool_calling_agent,AgentExecutor 
from langchain.agents.middleware import ToolCallLimitMiddleware, ModelCallLimitMiddleware
#from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool
from langchain_openai import ChatOpenAI
# from langchain import hub
# from langchain.prompts import ChatPromptTemplate
import sentence_transformers

from huggingface_hub import login
from rag_engine.retriever import rerank


"""
from transformers import pipeline
from transformers import BitsAndBytesConfig
from pinecone.exceptions import NotFoundException
from rag_engine.embeddings import embed_texts
"""


# 1. Configuration et Login
load_dotenv()

if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
else:
    print("⚠️ Attention : Clé LANGSMITH_API_KEY manquante. L'observabilité est désactivée.")

pinecone_api_key = os.getenv('PINECONE_API_KEY')
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')

openrouter_api_key = os.getenv('OPENAI_API_KEY') 
openrouter_base_url = os.getenv('OPENAI_BASE_URL')

if not openrouter_api_key or not openrouter_base_url:
    raise ValueError("Les variables OPENAI_API_KEY et OPENAI_BASE_URL (pointant vers OpenRouter) sont manquantes dans le fichier .env.")

login(token=huggingface_api_key)

# 2. Initialiser les Embeddings

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Chargement du modèle d'embedding sur : {device}")

# Charger le modèle d'encodage de texte BAAI/bge-small-en-v1.5 de HuggingFace
embedding = HuggingFaceEmbeddings(model_name="HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1",
                                  model_kwargs={'device': device}, # Pin the model to the GP
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
@tool
def smart_retrieval(query: str) -> str:
    # 1. Recherche large dans Pinecone (k=20 au lieu de 5)
    """Utilise cet outil pour trouver des informations précises dans les documents financiers indexés.
    Args:
        query: recherche des termes proches de la requête utilisateur.
        """
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

tools = [smart_retrieval]

# 5. LLM (Gemini 2.5 Flash via OpenRouter API)
llm = ChatOpenAI(
    # Le modèle OpenRouter pour Gemini 2.5 Flash
    model="amazon/nova-2-lite-v1:free", 
    temperature=0, 
    # Le base_url et l'API key sont cruciaux pour OpenRouter
    openai_api_key=openrouter_api_key,
    base_url=openrouter_base_url,
)

# 6. Création de l'Agent Tool Calling (STABLE)
system_prompt = (
    "Tu es un expert RAG. Ton objectif est de répondre aux questions en utilisant uniquement l'outil 'recherche_documents_financiers'. "
    "Si l'outil ne retourne aucune information pertinente ou si la question est hors contexte financier, réponds simplement 'Unknown'. "
    "Maintiens une réponse courte, précise et factuelle en français."
)

# # Le prompt Tool Calling est plus simple et plus stable que ReAct
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ]
# )

# L'agent Tool Calling est le plus stable pour cette tâche
search_agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

SCRIPTDOCTOR_AGENT_PROMPT = ("""
    # ROLE
    Tu es le "Financial Storyteller", un expert en communication stratégique spécialisé dans la vulgarisation financière. 
    Ta mission est de transformer des données financières brutes en scripts de discours captivants pour un public de non-financiers.
    
    # OBJECTIF
    Traduire la complexité comptable en vision stratégique narrative.
    
    # PUBLIC CIBLE
    Non-financiers (employés, clients, grand public). Ils veulent comprendre :
    1. La santé de l'entreprise.
    2. La direction stratégique.
    3. L'impact concret sur leur quotidien.
    
    # CONTRAINTE TEMPORELLE (IMPÉRATIF)
    **Le discours ne doit pas dépasser 3 minutes.**
    Cela correspond à environ **350 à 450 mots maximum**.
    Tu dois être synthétique, percutant et aller droit au but. Si le sujet est vaste, coupe les détails techniques pour garder l'essentiel.
    
    # DIRECTIVES DE RÉDACTION (STYLE ET TON)
    1. **Langage Parlé :** Phrases courtes. Voix active. Écris pour l'oreille.
    2. **Zéro Jargon :** Pas d'EBITDA, de CAPEX ou de BFR sans une analogie immédiate (ex: "Le carburant pour avancer" au lieu de "Trésorerie").
    3. **Règle du "1 Chiffre = 1 Impact" :** Sélectionne maximum 3 chiffres clés pour tout le discours.
    4. **Connexion Émotionnelle :** Utilise le "Nous" et le "Vous".
    
    # INSTRUCTIONS DE FORMATAGE (SCÉNOGRAPHIE)
    Inclus des didascalies pour guider l'orateur :
    - **[GRAS]** : Mots à accentuer vocalement.
    - `[PAUSE]` : Silence dramatique (compter 2-3 secondes).
    - `(Note : ...)` : Indication d'émotion ou de gestuelle.
    
    # FORMAT DE SORTIE (STRICT)
    Tu dois impérativement structurer ta réponse selon le modèle ci-dessous :
    
    ---
    **TITRE :** [Titre court et accrocheur résumant le message]
    **DURÉE ESTIMÉE :** ~[X] minutes
    **INTENTION :** [Ex: Rassurant, Mobilisateur, Célébration]
    **LES 3 POINTS CLÉS :**
    * [Point 1]
    * [Point 2]
    * [Point 3]
    
    **SCRIPT :**
    [Insérer ici le texte du discours avec les didascalies de mise en scène]
    ---
    
    # PROCESSUS DE RÉFLEXION (CHAIN OF THOUGHT)
    Avant de générer le format de sortie :
    1. Identifie le message unique le plus important.
    2. Sélectionne les données qui soutiennent ce message (jette le reste).
    3. Vérifie que le volume de texte tient dans les 400 mots.
    
    # EXEMPLES (FEW-SHOT LEARNING)
    
    **Exemple 1 (Contexte : Croissance)**
    *Entrée Superviseur :* "CA +15% YOY, lancement réussi produit X."
    *Ta Sortie :*
    **TITRE :** Notre pari gagnant
    **DURÉE ESTIMÉE :** ~1 min
    **INTENTION :** Fierté collective
    **LES 3 POINTS CLÉS :**
    * Succès du produit X
    * Croissance des ventes de 15%
    * Remerciement des équipes
    
    **SCRIPT :**
    (Grand sourire) Bonjour à tous.
    L'année dernière, nous avons fait un pari audacieux avec le produit X.
    Aujourd'hui, les chiffres sont tombés : nos ventes ont bondi de **15%**. [PAUSE]
    C'est la preuve que votre audace paie. Merci à tous.
    
    ---
    **DÉBUT DE LA TÂCHE**
    Attends les données financières."""
)

script_agent = create_agent(
    llm,
    system_prompt=SCRIPTDOCTOR_AGENT_PROMPT,
)

@tool
def search_doc(request: str) -> str:
    """Search database for relevant informations.
    
    Do this when the user wants get relevant financial informations about his request.

    Input: Natural language financial request (e.g., 'Quel est l'objectif du fond FCP ?')
    """
    result = search_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


@tool
def write_script(request: str) -> str:
    """Generates a compelling 3-minute speech script for non-financial audiences based on financial data.
    
    Use this tool when you need to communicate financial results, KPIs, or strategic shifts to employees, clients, or the general public. 
    It translates complex accounting terms into simple metaphors/stories and structures the output with stage directions (tone, pauses) for the speaker.
    
    Input: Raw financial documentation exerpt, context, and key metrics (e.g., "Revenue +20%, EBITDA negative due to investment").
    """
    result = script_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You write scripts of 3 mins speech script based on searched documentation."
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence."
)

supervisor_agent = create_agent(
    model=llm,
    tools=[search_doc, write_script],
    system_prompt=SUPERVISOR_PROMPT,
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=5,
            run_limit=5,
            exit_behavior="end",
        ),
        ToolCallLimitMiddleware(
            tool_name="search",
            thread_limit=5,
            run_limit=3,
        )
    ],
)

"""
# L'Executor
agent_executor = AgentExecutor(
    agent=search_agent, 
    tools=tools, 
    verbose=True,
    # Le Tool Calling gère naturellement ces erreurs, mais on peut laisser la sécurité
    handle_parsing_errors=True,
    max_iterations=5
)

# 7. Pipeline Principal
def rag_pipeline(query):
    try:
        response = agent_executor.invoke({"input": query})
        return response["output"]
    except Exception as e:
        return f"Erreur critique lors de l'exécution de l'agent : {str(e)}"


# 5 chargement du LLM
# ✅ Nouveau chargement optimisé
TOK, LLM_MODEL = load_llm()

from transformers import pipeline, StoppingCriteria, StoppingCriteriaList

#creation de la classe critères d'arrets
class StopOnString(StoppingCriteria):

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

"""