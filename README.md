Voici une proposition de `README.md` complet et structuré pour votre projet **RAG-docfin**.

Il met en avant la nature "Agentique" du projet, l'architecture technique (LangGraph/Pinecone) et la fonctionnalité clé de "Human-in-the-Loop".

---

# 📈 RAG-DocFin : Assistant Financier Agentique (HITL)

**RAG-DocFin** est un agent intelligent spécialisé dans l'analyse de documents financiers (DIC, rapports annuels, etc.). Il utilise une architecture **RAG (Retrieval-Augmented Generation)** orchestrée par **LangGraph** pour transformer des données brutes en livrables de communication stratégique (Scripts de discours ou Slides).

La particularité de ce projet est son approche **Human-in-the-Loop (HITL)** : l'agent ne se contente pas de générer du texte, il collabore avec l'utilisateur en demandant validation ou modification avant de produire des livrables critiques.

---

## ✨ Fonctionnalités Clés

* **🔍 Recherche Intelligente (RAG) :** Indexation et recherche sémantique dans des documents PDF financiers via **Pinecone** et des embeddings **HuggingFace**.
* **🤖 Architecture Multi-Agents (Superviseur) :** Un agent superviseur route les demandes vers des experts spécialisés :
* `search_doc` : Expert en recherche documentaire.
* `write_script` : "Financial Storyteller" qui rédige des discours de 3 minutes engageants.
* `create_slides` : Générateur de structure pour 3 slides synthétiques (Markdown).


* **✋ Human-in-the-Loop (HITL) :** Middleware d'interception qui permet à l'utilisateur de :
* Valider une action avant son exécution.
* Refuser une proposition et donner un feedback correctif.
* Changer de format à la volée (ex: passer d'un Script à des Slides).


* **🧠 Modèles Flexibles :** Compatible avec les modèles via **OpenRouter** (DeepSeek, GPT-OSS) ou OpenAI.

---

## 🛠️ Architecture Technique

Le projet repose sur la stack moderne suivante :

* **LangChain / LangGraph :** Orchestration du graphe d'états et gestion de la mémoire (Checkpointers).
* **Pinecone :** Base de données vectorielle pour le stockage des embeddings documentaires.
* **HuggingFace (`sentence-transformers`) :** Modèle d'embedding local ou API (`HIT-TMG/KaLM-embedding-multilingual...`).
* **LangSmith :** (Optionnel) Pour le tracing et l'observabilité des agents.

---

## 🚀 Installation

### 1. Prérequis

* Python 3.10 ou supérieur
* Compte Pinecone (API Key)
* Compte HuggingFace (Token)
* Clé API pour le LLM (OpenAI ou OpenRouter)

### 2. Cloner le projet

```bash
git clone https://github.com/votre-user/rag-docfin.git
cd rag-docfin

```

### 3. Environnement Virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

### 4. Configuration (.env)

Créez un fichier `.env` à la racine et remplissez-le avec vos clés :

```ini
# LLM Provider (OpenRouter ou OpenAI)
OPENAI_API_KEY=sk-or-votre-cle-openrouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1  # Si utilisation d'OpenRouter

# Vector Database
PINECONE_API_KEY=votre-cle-pinecone

# Embeddings
HUGGINGFACE_API_KEY=votre-token-hf

# Observabilité (Optionnel)
LANGSMITH_API_KEY=votre-cle-langsmith
LANGCHAIN_TRACING_V2=true

```

---

## 📖 Utilisation

### Ingestion des documents

Pour indexer vos PDF (situés dans le dossier `DIC/`) dans Pinecone :

```bash
python loadchunkembed.py

```

### Lancer l'Agent Interactif

Pour démarrer une session de chat avec supervision humaine :

```bash
python pipeline.py

```

### Exemple de flux d'interaction

1. **Utilisateur :** "Quels sont les frais du fonds FCP ?"
2. **Agent (Interruption) :** "Je vais lancer une recherche. Voulez-vous un Script ou des Slides ensuite ?"
3. **Utilisateur :** "Fais un Script."
4. **Agent :** *Effectue la recherche...*
5. **Agent (Interruption) :** "J'ai les infos. Je suis prêt à rédiger le script. [Valider/Rejeter] ?"
6. **Utilisateur :** "Valider."
7. **Agent :** *Génère le script final.*

---

## 📂 Structure du Projet

* `pipeline.py` : Cœur de l'application. Contient la logique des agents, le graphe LangGraph et la boucle d'interaction HITL.
* `loadchunkembed.py` : Script ETL pour charger, découper (chunking) et vectoriser les documents PDF.
* `rag_engine/` : Modules utilitaires pour le chargement des modèles et la configuration du retriever.
* `DIC/` : Dossier contenant les documents sources (PDFs).
* `dataset_eval/` : Jeux de données pour l'évaluation de la performance du RAG.

---

## 🤝 Contribution

Les Pull Requests sont les bienvenues. Pour les changements majeurs, veuillez d'abord ouvrir une issue pour discuter de ce que vous souhaitez modifier.

## 📄 Licence

[MIT](https://choosealicense.com/licenses/mit/)
