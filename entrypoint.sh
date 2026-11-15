#!/bin/sh
# Готовит базу знаний в локальном Qdrant при первом запуске, затем стартует веб-сервер.
set -e

if [ ! -d ./qdrant_chunks ]; then
    echo "Первый запуск: индексация базы знаний (data/kb.zip) в Qdrant..."
    python qdrant_split_base.py
fi

exec python app.py
