
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

# --- HITL ---
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

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
    model="openai/gpt-oss-20b:free", 
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

SLIDEGEN_AGENT_PROMPT = ("""
    # ROLE
    Tu es le **Financial Storyteller**, expert en communication stratégique spécialisé dans la vulgarisation financière.
    Ta mission : transformer **n’importe quel type d’informations financières** (entreprise, produit financier, indicateurs macro, portefeuille, FCP, performance, risque, allocation, etc.) en **3 slides claires** pour des non-financiers.

    # OBJECTIF
    Rendre des données financières compréhensibles **sans supposer le contexte** (entreprise, marché, produit, organisation).
    Tu ne dois **jamais extrapoler** la nature de l’entité analysée.

    # PUBLIC CIBLE
    Non-financiers (clients, épargnants, collaborateurs, grand public).  
    Ils cherchent à comprendre :
    1) La situation actuelle  
    2) La dynamique / les tendances  
    3) Les conséquences concrètes  

    # CONTRAINTE TEMPORELLE (IMPÉRATIF)
    Contenu lisible et présentable en **3 minutes maximum**.
    Volume cible : **3 slides × 40–70 mots** (≈ **150–200 mots max**).

    # DIRECTIVES DE RÉDACTION (STYLE ET TON)
    1) **Neutre et factuel :** pas de storytelling narratif, pas de ton corporate.
    2) **Zéro jargon implicite :** tout terme technique doit être simplifié ou reformulé.
    3) **Règle "1 chiffre = 1 impact" :** **3 chiffres maximum sur l’ensemble**.
    4) **Pas d’hypothèses :** tu reformules uniquement ce qui est fourni.

    # INSTRUCTIONS DE FORMATAGE (SLIDES)
    - Titres génériques, non sectoriels.
    - Bullets courts, lisibles à l’écran.
    - Aucun script oral, aucune didascalie.
    - Aucune projection non explicitement donnée.

    # FORMAT DE SORTIE (STRICT) — MARKDOWN
    Tu dois produire **exactement 3 slides**, structurées ainsi :

    ---

    ## Slide 1 — Situation actuelle
    **Ce que montrent les données :**
    - Point factuel principal
    - Point factuel secondaire

    **Chiffre clé (si fourni) :** … → **Ce que cela signifie concrètement :** …

    ---

    ## Slide 2 — Dynamique observée
    **Évolution ou tendance mise en évidence :**
    - Variation, comparaison ou stabilité
    - Élément explicatif explicitement présent dans les données

    **Chiffre clé (si fourni) :** … → **Lecture simple :** …

    ---

    ## Slide 3 — Implications concrètes
    **Ce que ces données impliquent :**
    - Conséquence directe et vérifiable
    - Point de vigilance ou d’opportunité (si mentionné dans les données)

    **Chiffre clé (si fourni) :** … → **Impact pratique :** …

    ---

    # RÈGLES STRICTES
    - Si une information n’est pas fournie → **ne pas l’inventer**.
    - Si un chiffre n’a pas d’impact clair → **ne pas l’utiliser**.
    - Si les données sont partielles → **le refléter explicitement**.

    # PROCESSUS DE RÉFLEXION (INVISIBLE DANS LA SORTIE)
    1) Reformuler les données **sans interprétation sectorielle**.
    2) Identifier faits → tendances → implications.
    3) Vérifier lisibilité écran et neutralité.
    4) Vérifier : **3 chiffres max**, aucun implicite.

    # EXEMPLE (ABSTRAIT)
    *Entrée superviseur :* "Performance +4,2% sur 12 mois, volatilité modérée, frais stables."

    ## Slide 1 — Situation actuelle
    **Ce que montrent les données :**
    - Performance positive sur la période observée
    - Niveau de risque décrit comme modéré

    **Chiffre clé :** +4,2% → **Cela indique :** une progression mesurée de la valeur.

    # DÉBUT DE LA TÂCHE
    Attends les données financières.
    Génère uniquement les **3 slides en markdown**, sans ajout de contexte.
"""
)

slide_agent = create_agent(
    llm,
    system_prompt=SLIDEGEN_AGENT_PROMPT,
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

@tool
def create_slides(request: str) -> str:
    """Generates 3 concise presentation slides (Markdown) for non-financial audiences based on financial data.

    Use this tool to communicate financial results, KPIs, performance, risk, or strategic signals (company, product, fund, portfolio, or macro data) to employees, clients, or the general public.
    It simplifies financial information without assuming context, avoids jargon, and highlights facts, trends, and concrete implications only.

    Output principles

    Exactly 3 slides, screen-readable, no script.

    Neutral and factual: no storytelling, no projections, no extrapolation.

    Maximum 3 key figures total, each linked to a clear, concrete meaning.

    Generic slide logic: current situation → observed dynamics → practical implications.

    Input
    Raw financial excerpts, context (if any), and key metrics
    (e.g. “Revenue +20%, negative EBITDA due to investment”, “FCP performance +4.2% over 12 months, moderate volatility”).
    """
    result = slide_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

# --- À INSÉRER AVANT LA CRÉATION DU SUPERVISOR_AGENT ---

# 1. Configuration de la mémoire (Obligatoire pour le HITL)
# Permet de sauvegarder l'état de l'agent pendant qu'il attend la validation humaine
memory = MemorySaver()

# 2. Configuration du Middleware HITL
# On demande une interruption explicite avant l'exécution de l'outil "write_script"
hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "write_script": True,
        "create_slides": True
    }
)

SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "Based on the user's request, you write scripts of 3 mins speech or create slides based on searched documentation."
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence."
)

supervisor_agent = create_agent(
    model=llm,
    tools=[search_doc, write_script,create_slides],
    system_prompt=SUPERVISOR_PROMPT,
    middleware=[
        hitl_middleware,
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
    checkpointer=memory
)

# --- REMPLACEMENT DE LA FONCTION D'EXÉCUTION ---

def run_interactive_pipeline(user_query: str, thread_id: str = "session_default"):
    """
    Exécute le pipeline RAG avec supervision humaine.
    Permet de valider ou de changer le format (Slide <-> Script) à la volée.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n🚀 Démarrage : '{user_query}'")
    
    # 1. Lancement initial
    result = supervisor_agent.invoke(
        {"messages": [("user", user_query)]},
        config=config
    )

    # 2. Vérification de l'interruption
    snapshot = supervisor_agent.get_state(config)
    
    if snapshot.next:
        # Récupération de l'action proposée
        last_msg = snapshot.values["messages"][-1]
        
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            tool_call = last_msg.tool_calls[0]
            tool_name = tool_call['name']
            
            # Cas A : L'agent veut faire des SLIDES
            if tool_name == "create_slides":
                print(f"\n📊 PROPOSITION : L'agent veut générer des SLIDES.")
                user_input = input("Choix : [V]alider, [C]hanger pour un Script, [A]nnuler : ").lower()
                
                if user_input == 'c':
                    print("🔄 Bascule vers le mode SCRIPT...")
                    # On force le changement d'intention
                    supervisor_agent.update_state(config, {"messages": [{
                        "role": "user", 
                        "content": "Change de format : je ne veux pas de slides, rédige-moi un SCRIPT de discours à la place."
                    }]})
                    result = supervisor_agent.invoke(None, config=config)
                
                elif user_input == 'v':
                    print("✅ Validation des Slides...")
                    result = supervisor_agent.invoke(Command(resume="approve"), config=config)

            # Cas B : L'agent veut faire un SCRIPT
            elif tool_name == "write_script":
                print(f"\n🎙️ PROPOSITION : L'agent veut rédiger un SCRIPT.")
                user_input = input("Choix : [V]alider, [C]hanger pour des Slides, [A]nnuler : ").lower()
                
                if user_input == 'c':
                    print("🔄 Bascule vers le mode SLIDES...")
                    # On force le changement d'intention
                    supervisor_agent.update_state(config, {"messages": [{
                        "role": "user", 
                        "content": "Change de format : je ne veux pas de discours, fais-moi 3 SLIDES synthétiques à la place."
                    }]})
                    result = supervisor_agent.invoke(None, config=config)
                
                elif user_input == 'v':
                    print("✅ Validation du Script...")
                    result = supervisor_agent.invoke(Command(resume={"action": "approve"}), config=config)

    # 3. Retour du résultat final
    if result and "messages" in result:
        return result["messages"][-1].content
    return "Aucune réponse générée."