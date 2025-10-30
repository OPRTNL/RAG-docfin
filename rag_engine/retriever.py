from typing import List
import numpy as np
from sentence_transformers import CrossEncoder

_RERANK = None
def _get_reranker():
    global _RERANK
    if _RERANK is None:
        _RERANK = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANK

def rerank(query: str, docs: List, use_rerank: bool = False):
    if not use_rerank:
        return docs
    ce = _get_reranker()
    pairs = [(query, getattr(d, "text", getattr(d, "page_content", ""))) for d in docs]
    scores = ce.predict(pairs)
    order = np.argsort(-scores)
    return [docs[i] for i in order]
