import glob
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException


#load des documents
loader = PyPDFDirectoryLoader("DIC")
documents = loader.load()

print('nombre de pages :', len(documents))

#chargement des variable d'env
load_dotenv()


pinecone_api_key = os.getenv('PINECONE_API_KEY')
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')

print('nombre de docs', len([f for f in os.listdir("./DIC/")]))



# Charger le modèle d'encodage de texte BAAI/bge-small-en-v1.5 de HuggingFace
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings" : True})

text_splitter = SemanticChunker(embedding)

# Division du document en morceaux (chunks)
chunks = text_splitter.split_documents(documents=documents)


# Affichage du nombre de morceaux créés à partir du document PDF
print(f"{len(chunks)} chunks ont été créés par le splitter à partir du document PDF.")



# TODO: Inscrire la clé API Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)

# effacage de la base Pinecone et recréation d'un index vide
pinecone.delete_index("rag")
pinecone.create_index("rag", dimension=1024,spec=ServerlessSpec(cloud="aws", region="us-east-1"))

from langchain_pinecone.vectorstores import PineconeVectorStore

# Initialiser le VectorStore de LlamaIndex avec l'index de Pinecone
pinecone_index = pinecone.Index("rag")
vector_store = PineconeVectorStore(
    index=pinecone_index,
    embedding=embedding
)

# remplissage de l'index pinecone avec les vecteurs
add_result = vector_store.add_documents(chunks)
print(f"{len(add_result)} vecteurs ont été ajoutés dans Pinecone.")








