"""Минимальный пример сабмита ПСБ: embeddings + rerank + LLM.

Это учебный каркас, а не полноценное RAG-решение: используется только
простой поиск по словам среди документов и короткие текстовые фрагменты.
"""

from __future__ import annotations

import math
import sys
import time

import httpx

import config
import utils
from schemas import Question, Document


def post(client: httpx.Client, path: str, key: str, payload: dict) -> dict:
    if not key:
        raise ValueError("На платформе отсутствует API-ключ")
    
    response = client.post(
        config.API_BASE + path,
        headers={"Authorization": "Bearer " + key},
        json=payload,
    )
    response.raise_for_status()
    
    return response.json()


def embeddings(client: httpx.Client, texts: list[str]) -> list[list[float]]:
    """Пример вызова EMB_MODEL через POST /embeddings."""
    result = post(client, "/embeddings", config.EMBED_KEY, {
        "model": config.EMB_MODEL,
        "input": texts,
        "encoding_format": "float",
    })

    vectors: list[list[float] | None] = [None] * len(texts)
    for position, item in enumerate(result["data"]):
        index = int(item.get("index", position))
    
        if index < 0 or index >= len(texts):
            raise ValueError("Некорректный индекс embedding")
    
        vector = item["embedding"]
        norm = math.sqrt(sum(float(x) ** 2 for x in vector)) or 1.0
        vectors[index] = [float(x) / norm for x in vector]
    
    if any(v is None for v in vectors):
        raise ValueError("Получены не все embeddings")
    
    return vectors  # type: ignore[return-value]


def rerank(client: httpx.Client, question: Question, candidates: list[Document]) -> list[int]:
    """Пример вызова RERANKER_MODEL через POST /rerank."""
    result = post(client, "/rerank", config.RERANK_KEY, {
        "model": config.RERANKER_MODEL,
        "query": question.content,
        "documents": [candidate.name + "\n" + utils.snippet(candidate.text) for candidate in candidates],
        "top_n": min(config.RERANKER_TOP_N, len(candidates)),
    })
    
    items = result.get("results", result.get("data", []))
    indices = [int(item["index"]) for item in items]
    
    return [i for i in indices if 0 <= i < len(candidates)][:config.RERANKER_TOP_N]


def select_documents(client: httpx.Client, question: Question,
                     documents: list[Document]) -> list[Document]:
    # Простая предварительная выборка по совпадению слов; не полноценный RAG.
    query_words = utils.words(question.content)
    candidates = sorted(
        documents,
        key=lambda d: len(query_words & utils.words(d.name + " " + d.text)),
        reverse=True,
    )[:config.TOP_K]

    # Шаг 1: демонстрация использования эмбеддера.
    try:
        vectors = embeddings(client, [question.content] + [utils.snippet(document.text) for document in candidates])
        query_vec = vectors[0]
        
        scores = [sum(a * b for a, b in zip(query_vec, vec)) for vec in vectors[1:]]
        candidates = [candidates[i] for i in sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)]
    
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        print(f"Embeddings недоступны: {type(exc).__name__}; используем поиск по словам", file=sys.stderr)

    # Шаг 2: демонстрация реранкера. При ошибке продолжаем без него.
    try:
        indices = rerank(client, question, candidates)
        if indices:
            return [candidates[i] for i in indices]
    
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        print(f"Reranker недоступен: {type(exc).__name__}; используем текущий порядок", file=sys.stderr)
    
    return candidates[:config.RERANKER_TOP_N]


def generate(client: httpx.Client, question: Question, documents: list[Document]) -> str:
    """Пример вызова LLM_MODEL через POST /chat/completions."""
    context = "\n\n".join(document.name + "\n" + utils.snippet(document.text) for document in documents)
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Ответь на вопрос по приложенным фрагментам. Не выдумывай юридические нормы. Если ответа нет, сообщи об этом."},
            {"role": "user", "content": "Фрагменты:\n" + context + "\n\nВопрос:\n" + question.content},
        ],
        "temperature": config.TEMPERATURE,
        "max_tokens": config.MAX_TOKENS,
    }
    
    for attempt in range(2):
        try:
            result = post(client, "/chat/completions", config.LLM_KEY, payload)
            content = result["choices"][0]["message"].get("content", "")
            if isinstance(content, str) and content.strip():
                return " ".join(content.split())  # одна физическая строка CSV
            raise ValueError("Модель вернула пустой ответ")
    
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            print(f"LLM attempt {attempt + 1}: {type(exc).__name__}", file=sys.stderr)
            if attempt == 0:
                time.sleep(1)
    
    return config.FALLBACK


def main() -> None:
    if not config.API_BASE or not config.LLM_KEY:
        raise ValueError("Нужны LLM_BASE_URL и LLM_API_KEY (или их аналоги)")
    
    questions = utils.load_questions(utils.find_file("questions.csv"))
    documents = utils.load_documents(utils.find_file("kb.zip", "knowledge-base.zip", "knowledge_base.zip"))
    
    print(f"Вопросы: {len(questions)}; документы: {len(documents)}; TOP_K={config.TOP_K}", flush=True)

    def work(position: int, question: Question) -> tuple[int, str]:
        try:
            with httpx.Client(timeout=90.0) as client:
                selected = select_documents(client, question, documents)
                return position, generate(client, question, selected)
       
        except Exception as exc:
            print(f"Ошибка вопроса {position}: {type(exc).__name__}", file=sys.stderr)
            return position, config.FALLBACK

    answers = utils.run_workers(questions, work)

    utils.write_submission_file(questions, answers)


if __name__ == "__main__":
    main()
