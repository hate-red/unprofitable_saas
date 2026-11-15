"""Минимальный веб-слой поверх RAG-пайплайна из solution.py.

Frontend отдаёт статику из frontend/, API:
  POST /api/chat  {"message": "..."} -> {"answer": "..."}
  GET  /api/health
"""
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from schemas import Question
from solution import select_documents, generate


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.API_BASE or not config.LLM_KEY:
        raise RuntimeError("Нужны LLM_BASE_URL и LLM_API_KEY (или их аналоги)")

    # Один общий клиент на всё приложение.
    app.state.client = httpx.Client(timeout=90.0)
    app.state.system_prompt = config.system_prompt_path.read_text(encoding="utf-8")
    app.state.user_prompt = config.user_prompt_path.read_text(encoding="utf-8")

    try:
        yield
    finally:
        app.state.client.close()


app = FastAPI(title="RAG demo", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class Source(BaseModel):
    source: str
    doc_type: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = Question(id="web", content=request.message.strip())
    if not question.content:
        raise HTTPException(400, "Пустое сообщение")

    documents = select_documents(app.state.client, question)
    answer = generate(
        client=app.state.client,
        question=question,
        documents=documents,
        system_prompt=app.state.system_prompt,
        user_prompt=app.state.user_prompt,
    )
    sources = [
        Source(source=document.metadata.get("source", str(document.path)),
               doc_type=document.metadata.get("doc_type", ""))
        for document in documents
    ]
    return ChatResponse(answer=answer, sources=sources)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    # Онлайн-сервер: база в Qdrant должна быть уже подготовлена (см. entrypoint.sh).
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(config.env("PORT", "8000")))
