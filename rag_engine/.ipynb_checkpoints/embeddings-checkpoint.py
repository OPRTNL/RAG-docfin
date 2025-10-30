from sentence_transformers import SentenceTransformer
from joblib import Memory
import numpy as np
import torch

mem = Memory(".cache/joblib", verbose=0)
EMB_ID = "sentence-transformers/all-MiniLM-L6-v2"

@mem.cache
def embed_texts(texts):
    model = SentenceTransformer(EMB_ID, device="cuda" if torch.cuda.is_available() else "cpu")
    return np.array(model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False))
