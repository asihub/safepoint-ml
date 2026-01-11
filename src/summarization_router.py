"""
summarization_router.py — FastAPI endpoint for wellbeing resource summarization.
Add to main.py: app.include_router(summarization_router)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from .summarization import summarize_url

router = APIRouter(prefix="/summarize", tags=["summarization"])


class SummarizeRequest(BaseModel):
    url: HttpUrl


class SummarizeResponse(BaseModel):
    url:     str
    excerpt: str | None


@router.post("", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Fetch a wellbeing resource URL, extract article text,
    and return a short AI-generated summary.
    """
    url = str(request.url)
    excerpt = summarize_url(url)

    if excerpt is None:
        raise HTTPException(
            status_code=422,
            detail="Could not extract or summarize content from the provided URL."
        )

    return SummarizeResponse(url=url, excerpt=excerpt)
