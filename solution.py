import sys
import time

import httpx

import config
import utils

from schemas import Question, Document
from qdrant_split_base import (
    setup_qdrant,
    search_qdrant_vector,
    search_qdrant_text,
    search_qdrant_hybrid,
)


def rerank(client: httpx.Client, question: Question, candidates: list[Document]) -> list[int]:
    """Пример вызова RERANKER_MODEL через POST /rerank."""
    result = utils.post(client, "/rerank", config.RERANK_KEY, {
        "model": config.RERANKER_MODEL,
        "query": question.content,
        "documents": [candidate.content for candidate in candidates], 
        "top_n": min(config.RERANKER_TOP_N, len(candidates)),
    })
    
    items = result.get("results", result.get("data", []))
    indices = [int(item["index"]) for item in items]
    
    return [i for i in indices if 0 <= i < len(candidates)][:config.RERANKER_TOP_N]


def select_documents(client: httpx.Client, question: Question) -> list[Document]:
    # Step 1: DB search.
    try:
        candidates = search_qdrant_hybrid(question)
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        print(f"Embeddings недоступны: {type(exc).__name__}; используем поиск по словам", file=sys.stderr)

    # Step 2: Reranker.
    try:
        indices = rerank(client, question, candidates) # type: ignore
        if indices:
            return [candidates[i] for i in indices] # type: ignore

    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        print(f"Reranker недоступен: {type(exc).__name__}; используем текущий порядок", file=sys.stderr)
    
    return candidates[:config.RERANKER_TOP_N] # type: ignore


def generate(
    client: httpx.Client, 
    question: Question, 
    documents: list[Document],
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Пример вызова LLM_MODEL через POST /chat/completions."""
    context = "\n\n".join(document.metadata['source'] + "\n" + document.content for document in documents)
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(question=question, context=context)},
        ],
        "temperature": config.TEMPERATURE,
        "max_tokens": config.MAX_TOKENS,
    }
    
    for attempt in range(2):
        try:
            result = utils.post(client, "/chat/completions", config.LLM_KEY, payload)
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
    
    chunks_per_question = []
    with httpx.Client(timeout=90.0) as client:
        for question in questions:
            chunks = select_documents(client, question)
            chunks_per_question.append(chunks)

    system_prompt = config.system_prompt_path.read_text()
    user_prompt = config.user_prompt_path.read_text()
    
    print(f"Вопросы: {len(questions)}; документы: {len(documents)}; TOP_K={config.TOP_K}", flush=True)

    def work(position: int, question: Question, chunks: list[Document]) -> tuple[int, str]:
        try:
            with httpx.Client(timeout=90.0) as client:
                return (
                    position, 
                    generate(
                        client=client, 
                        question=question, 
                        documents=chunks,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt
                    )
                )
        
        except Exception as exc:
            print(f"Ошибка вопроса {position}: {type(exc).__name__}", file=sys.stderr)
            return position, config.FALLBACK

    answers = utils.run_workers(questions, chunks_per_question, work)
    utils.write_submission_file(questions, answers)


if __name__ == "__main__":
    setup_qdrant()
    main()
