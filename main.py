import os
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import ml_service

app = FastAPI(
    title="Scientific Articles Recommendation API",
    description="Система семантического поиска и LLM-суммаризации научных публикаций",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=500, description="Текст запроса для поиска")
    top_k: int = Field(default=2, ge=1, le=10, description="Количество рекомендуемых статей")

class RecommendationResponse(BaseModel):
    similarity: float
    title: str
    abstract: str
    summary: str

@app.post("/recommend", response_model=List[RecommendationResponse])
def get_recommendations(request: QueryRequest):
    try:
        results = ml_service.find_similar_articles(request.text, request.top_k)
        response = []
        for res in results:
            summary = ml_service.summarize_text(res["abstract"])
            response.append(
                RecommendationResponse(
                    similarity=res["similarity"],
                    title=res["title"],
                    abstract=res["abstract"],
                    summary=summary,
                )
            )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": ml_service.index is not None}
