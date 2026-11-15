# RAG-ассистент: веб-демо поверх скрипта для платформы

Минимальный frontend + backend над существующим RAG-пайплайном (`solution.py`,
`qdrant_split_base.py`, `splitter.py`, `utils.py` — ядро не изменено, кроме двух
исправлений багов, см. ниже). Никакой БД: база знаний индексируется в локальный
Qdrant (файловый режим), ответы генерируются через внешний LLM API.

## Структура

- `app.py` — FastAPI-backend: `POST /api/chat` (гибридный поиск → реранк → LLM), отдаёт статику frontend.
- `frontend/` — одна экранная страница чата (HTML/CSS/JS, без сборки и зависимостей).
- `entrypoint.sh` — при первом запуске строит индекс из `data/kb.zip`, затем запускает сервер.
- `Dockerfile`, `docker-compose.yml` — всё поднимается одной командой.

## Исправленные баги в основном скрипте

1. `solution.py`: результат гибридного поиска не зависел от `TOP_K` — теперь передаётся `limit=config.TOP_K`.
2. `utils.py` (`run_workers`): в `work` передавался `question.content` (строка) вместо объекта `Question`, из-за чего `select_documents` падал (`AttributeError`). Теперь передаётся `Question`.

## Запуск (одна команда)

Требуется Docker. Положите файлы задачи в `./data` (`kb.zip` — база знаний;
опционально `questions.csv` для пакетного режима платформы).

```bash
LLM_BASE_URL=https://api.openai.com/v1 LLM_API_KEY=sk-... docker compose up --build
```

Переменные можно так же записать в файл `.env` рядом с `docker-compose.yml`
(Docker Compose подхватит его сам):

```
LLM_BASE_URL=https://...
LLM_API_KEY=sk-...
```

После старта откройте **http://localhost:8000** — страница чата.
Первый запуск индексирует базу знаний (нужен доступ к `/embeddings`),
дальше индекс переиспользуется из volume `qdrant-data`.

## Настройки (необязательные переменные окружения)

Те же, что описаны в исходном README платформы: `EMBEDDER_API_KEY`,
`RERANKER_API_KEY`, `LLM_MODEL`, `EMB_MODEL`, `RERANKER_MODEL`, `TOP_K`,
`RERANKER_TOP_N`, `TEMPERATURE`, `MAX_TOKENS`, `PORT` (порт веб-сервера, по умолчанию 8000).

## API

- `POST /api/chat` — тело `{"message": "текст вопроса"}`,
  ответ `{"answer": "...", "sources": [{"source": "...", "doc_type": "..."}]}`.
- `GET /api/health` — проверка живости.

## Пакетный режим (для платформы)

Работает как раньше: `python qdrant_split_base.py && python solution.py`
создаёт `submission.csv` из `data/questions.csv`.
