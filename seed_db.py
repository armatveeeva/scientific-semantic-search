import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def seed_database():
    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    articles = [
        {
            "title": "Применение градиентного бустинга в материаловедении",
            "abstract": "В данной работе исследуется использование XGBoost для предсказания механических свойств новых сплавов на основе титана.",
            "authors": "Иванов И.И., Петров П.П.",
            "year": 2023,
        },
        {
            "title": "Глубокое обучение для анализа микроструктуры стали",
            "abstract": "Предложен новый подход на основе сверточных нейронных сетей (CNN) для автоматической классификации дефектов в металлических образцах.",
            "authors": "Сидоров С.С.",
            "year": 2024,
        },
        {
            "title": "Обзор методов NLP в обработке научных текстов",
            "abstract": "Статья посвящена применению моделей трансформеров и извлечению именованных сущностей (NER) из химических и физических публикаций.",
            "authors": "Смирнова А.А.",
            "year": 2023,
        },
    ]

    embeddings = []
    for article in articles:
        emb = model.encode(article["abstract"])
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)

    embeddings_array = np.array(embeddings).astype("float32")
    dimension = embeddings_array.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)

    faiss.write_index(index, "articles_faiss.index")
    with open("articles_metadata.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Successfully created FAISS index and metadata for {len(articles)} articles."
    )


if __name__ == "__main__":
    seed_database()
