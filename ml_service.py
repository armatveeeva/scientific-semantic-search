import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

logger.info("Загрузка ML-моделей и FAISS индекса...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", trust_remote_code=True)

try:
    index = faiss.read_index("articles_faiss.index")
    with open("articles_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info("Все компоненты загружены успешно!")
except FileNotFoundError:
    logger.warning("FAISS индекс или метаданные не найдены. Сначала запустите seed_db.py")
    index = None
    metadata = []


def find_similar_articles(query_text: str, top_k: int = 2):
    if index is None:
        return []
    query_emb = embedding_model.encode(query_text)
    query_emb = query_emb / np.linalg.norm(query_emb)
    query_array = np.array([query_emb]).astype("float32")
    distances, indices = index.search(query_array, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append({
                "similarity": float(distances[0][i]),
                "title": metadata[idx]["title"],
                "abstract": metadata[idx]["abstract"],
                "authors": metadata[idx]["authors"],
                "year": metadata[idx]["year"],
            })
    return results


def summarize_text(text: str) -> str:
    """LLM-суммаризация с защитой от сбоев"""
    try:
        max_length = min(len(text), 1024)
        summary = summarizer_pipeline(text[:max_length], max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"Ошибка LLM-суммаризации: {e}. Возвращаем начало текста как fallback.")
        return text.split(".")[0] + "."
