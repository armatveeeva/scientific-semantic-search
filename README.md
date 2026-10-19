# Обработка и структурирование научных публикаций в области материаловедения

Система семантического поиска и рекомендаций научных публикаций с использованием **векторных баз данных (FAISS)** и **NLP-суммаризации**.

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)](https://github.com/facebookresearch/faiss)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Проект демонстрирует построение **production-ready ML-сервиса** для семантического поиска научных статей. Система принимает текстовый запрос, находит наиболее релевантные публикации с помощью векторного поиска (FAISS) и генерирует краткую выжимку с использованием NLP-алгоритмов.

## Описание предметной области

В научно-исследовательских и инженерных организациях ежедневно публикуются тысячи новых научных статей, технических отчетов и патентов. 

Традиционный поиск по ключевым словам часто неэффективен. Он не учитывает семантический смысл, синонимы и контекст, заставляя исследователей тратить огромное количество времени на ручной просмотр нерелевантных документов.

## Технологический стек

- **Язык:** Python 3.11+
- **Backend & API:** FastAPI (асинхронный фреймворк), Uvicorn (ASGI-сервер), Pydantic (валидация данных)
- **Тестирование:** Pytest + httpx (интеграционные тесты API)
- **Векторизация текста:** Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Векторный поиск:** FAISS (`faiss-cpu`) — библиотека от Meta для поиска похожих векторов
- **NLP & Суммаризация:** Sumy (алгоритм TextRank для экстрактивной суммаризации)
- **Хранение данных:** SQLite + JSON (метаданные статей)
- **CI/CD:** GitHub Actions (автоматический пайплайн тестирования)

## Архитектура

```mermaid
flowchart TB
    subgraph Client["Client"]
        UI[Frontend / Postman / curl]
    end

    subgraph API["API Layer (FastAPI)"]
        Router[Router<br/>/recommend]
        Validator[Pydantic Validator<br/>QueryRequest]
    end

    subgraph ML["ML Service Layer"]
        Encoder[SentenceTransformer<br/>paraphrase-multilingual-MiniLM]
        Search[FAISS Vector Search]
        Summarizer[LLM Pipeline<br/>Hugging Face]
    end

    subgraph Data["Data Layer"]
        Index[(articles_faiss.index<br/>vectors)]
        Meta[(articles_metadata.json<br/>metadata)]
    end

    UI -->|HTTP POST| Router
    Router --> Validator
    Validator --> Encoder
    Encoder -->|query vector| Search
    Search -->|read| Index
    Search -->|read| Meta
    Search -->|top-k results| Summarizer
    Summarizer -->|summary| Router
    Router -->|JSON| UI

    style Client fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style API fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style ML fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style Data fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
   ```

## Быстрый старт

# 1. Клонирование и переход в директорию (если еще не сделали)
# git clone https://github.com/armatveeeva/scientific-semantic-search.git
# cd scientific-semantic-search

# 2. Установка зависимостей через Poetry
poetry install

# 3. Активация виртуального окружения
poetry shell

# 4. Инициализация базы данных (создание FAISS индекса и метаданных)
python seed_db.py

# 5. Запуск сервера FastAPI в режиме разработки
uvicorn main:app --reload