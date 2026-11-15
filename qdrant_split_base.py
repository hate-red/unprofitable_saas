from qdrant_client import QdrantClient
from qdrant_client import models

import httpx
import math
import hashlib
from contextlib import closing
from itertools import chain

import config
import utils
import splitter    
from schemas import Question, Document


COLLECTION = "kb_chunks"
EMB_SIZE = 1024
QDRANT_PATH = "./qdrant_chunks"


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Gets embeddings for given texts"""
    with httpx.Client(timeout=90.0) as client:
        result = utils.post(client, "/embeddings", config.EMBED_KEY, {
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
    
    return vectors  # type: ignore


#Использую формат pointstruct, не вижу смысла колонками, данных всего ничего
def chunks_to_points(chunks: list[Document]) -> list[models.PointStruct]:
    """Convert document chunks into Qdrant points.

    Args:
        chunks: Documents to embed and convert.

    Returns:
        Qdrant points containing each chunk's ID, vector, and payload.
    """
    points: list[models.PointStruct] = []

    vectors = get_embeddings([chunk.content for chunk in chunks])

    for i, chunk in enumerate(chunks):
        payload = chunk.to_dict()

        point_id = hashlib.md5(
            f"{chunk.metadata['source']}::{chunk.content}".encode("utf-8")
        ).hexdigest()

        points.append(models.PointStruct(id=point_id, vector=vectors[i], payload=payload))

    return points


def save_to_qdrant(
    points: list[models.PointStruct], 
    batch_size: int = 64, 
    rebuild: bool = False, 
) -> int:
    """Create or update the Qdrant collection and upsert points in batches.

    Args:
        points: Points to upsert into the collection.
        batch_size: Maximum number of points to upsert per batch.
        rebuild: Whether to delete and recreate an existing collection.

    Returns:
        The total number of points in the collection after the upserts.
    """
    with closing(QdrantClient(path=QDRANT_PATH)) as client:
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION in existing and rebuild:
            print(f"Удаляем коллекцию {COLLECTION}", flush=True)
            client.delete_collection(collection_name=COLLECTION)
            existing.remove(COLLECTION)

        if COLLECTION not in existing:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(size=EMB_SIZE, distance=models.Distance.COSINE),
            )

            client.create_payload_index(
                collection_name=COLLECTION,
                field_name="metadata",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

            client.create_payload_index(
                collection_name=COLLECTION,
                field_name="content",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    lowercase=True,
                    scoring=True,
                ),
            )

        total = len(points)
        for i in range(0, total, batch_size):
            client.upsert(
                collection_name=COLLECTION,
                points=points[i : i + batch_size],
            )
            print(f"  загружено {min(i + batch_size, total)}/{total}", flush=True)

        count = client.count(collection_name=COLLECTION).count
        return count


def view_qdrant(
    limit: int = 5,
    show_vector: bool = False,
    preview_chars: int = 200,
) -> None:
    """Print a summary and a preview of points in the Qdrant collection.

    Args:
        limit: Maximum number of points to display.
        show_vector: Whether to request and display each point's vector.
        preview_chars: Maximum number of content characters to display per
            point.
    """
    with closing(QdrantClient(path=QDRANT_PATH)) as client:
        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION not in existing:
            print(f"Коллекция {COLLECTION} не найдена", flush=True)
            client.close()
            return

        total = client.count(collection_name=COLLECTION).count
        print(f"Всего точек в {COLLECTION}: {total}\n", flush=True)

        points, _ = client.scroll(
            collection_name=COLLECTION,
            limit=limit,
            with_payload=True,
            with_vectors=show_vector,
        )

    for i, p in enumerate(points, 1):
        print(f"--- [{i}] id={p.id} ---")
        if show_vector and isinstance(p.vector, list):
            print(f"vector: len={len(p.vector)}, первые 5={p.vector[:5]}")

        payload = p.payload

        print(f"metadata: {payload['metadata']}") # type: ignore
        print(f"content[:{preview_chars}]: {payload['content'][:preview_chars]}...") # type: ignore


def search_qdrant_vector(
    question: Question,
    limit: int = 5,
    score_threshold: float | None = None,
) -> list[Document]:
    """Search the collection using a vector embedding of the question.

    Args:
        question: `Question` to embed and use as the vector query.
        limit: Maximum number of matching points to return.
        score_threshold: Optional minimum score for returned points.

    Returns:
        `Document` objects.
    """
    with closing(QdrantClient(path=QDRANT_PATH)) as client:
        query_vector = get_embeddings([question.content])[0]

        result = client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )
        chunks = [Document(**point.payload) for point in result.points] # type: ignore
        return chunks


def search_qdrant_text(question: Question, limit: int = 5) -> list[Document]:
    """Search the collection using its indexed text content.

    Args:
        question: `Question` query to match against the ``content`` field.
        limit: Maximum number of matching points to return.

    Returns:
        `Document` objects.
    """
    with closing(QdrantClient(path=QDRANT_PATH)) as client:
        result = client.query_points(
            collection_name=COLLECTION,
            query=models.TextQuery(text=question.content),
            using="content",
            limit=limit,
            with_payload=True,
        )
        chunks = [Document(**point.payload) for point in result.points] # type: ignore
        return chunks


def search_qdrant_hybrid(
    question: Question, 
    limit: int = 5, 
    prefetch_limit: int = 20
) -> list[Document]:
    """Search using text and vector queries, combining results with RRF.

    Args:
        question: `Question` used for both the text query and vector embedding.
        limit: Maximum number of fused results to return.
        prefetch_limit: Maximum number of candidates to fetch for each query
            before fusion.

    Returns:
        Fused `Document` objects.
    """
    dense_vector = get_embeddings([question.content])[0]

    with closing(QdrantClient(path=QDRANT_PATH)) as client:
        result = client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=models.TextQuery(text=question.content),
                    using="content",
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=dense_vector,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        chunks = [Document(**point.payload) for point in result.points] # type: ignore
        return chunks


def setup_qdrant():
    documents = utils.load_documents(utils.find_file("kb.zip", "knowledge-base.zip", "knowledge_base.zip"))
    chunks = list(chain.from_iterable(splitter.split_document(doc) for doc in documents))

    print("== 1. Преобразование в векторы ==", flush=True)
    points = chunks_to_points(chunks)
    print(f"Готово: {len(points)} models.PointStruct, dim={len(points[0].vector)}\n", flush=True) # type: ignore

    print("== 2. Сохранение в Qdrant ==", flush=True)
    count = save_to_qdrant(points, rebuild=True)
    print(f"В коллекции: {count} точек\n", flush=True)

    print("== 3. Просмотр из Qdrant ==", flush=True)
    view_qdrant(limit=2, show_vector=True)


if __name__ == '__main__':
    setup_qdrant()

    print("Поиск по Qdrant ==", flush=True)
    question = Question(id='1', content='Какая арендная плата по договору с ООО «Северный Склад» и до какого числа её нужно вносить?')
    results = search_qdrant_text(question, limit=3)

    for match in results:
        print(match)
        print('\n\n')
