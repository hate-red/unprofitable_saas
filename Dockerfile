FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ (questions.csv, kb.zip) монтируется томом; qdrant_chunks/ — кэш базы знаний.
EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
