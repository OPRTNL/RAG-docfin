# RAG-docfin

## Description  
RAG-docfin est un pipeline Python conçu pour exploiter la technique de Retrieval-Augmented Generation (RAG) sur un corpus financier / documentaire. Il permet de :  
- charger [chunker/embedder] des documents (PDF, TXT, etc)  
- construire un index de vecteurs  
- interroger le système pour générer des réponses contextualisées via un LLM  
- évaluer les performances du système via des notebooks et scripts d’évaluation  

## Fonctionnalités principales  
- Pré-traitement des documents : découpage, nettoyage, embeddings  
- Construction d’un pipeline « ingestion → indexation → requête → réponse » (cf. `pipeline.py`)  
- Mode évaluation automatisé (cf. `eval.py`, `dataset_eval/`)  
- Notebooks pour expérimenter : `RAG.ipynb`, `RAG Self query.ipynb`  
- Script de chargement des chunks & embeddings : `loadchunkembed.py`  
- Gestion des dépendances via `requirements.txt`  

## Structure du dépôt  
