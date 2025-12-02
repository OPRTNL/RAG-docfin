# test_app.py
import streamlit as st
import sys
import os

# Récupère le chemin absolu du dossier parent (remonter d'un niveau)
parent_dir = os.path.abspath("..")

# Ajoute ce chemin à sys.path s'il n'y est pas déjà
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Vous pouvez maintenant faire votre import normalement
from pipeline import rag_pipeline, llm

st.title("RAG DOCFIN !")
st.write("Si tu vois ceci, ton interface fonctionne.")

name = st.text_input("Ta question :")

if name:
    answer = rag_pipeline(name)
    st.success(f"Réponse {answer}")

number = st.slider("Choisis un nombre :", 0, 100, 50)
st.write("Valeur sélectionnée :", number)