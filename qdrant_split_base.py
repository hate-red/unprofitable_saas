import hashlib
from pathlib import Path
import zipfile

from itertools import chain
from langchain_core.documents import Document as LCDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import re
from dataclasses import dataclass

@dataclass(slots=True)
class Question:
    id: str
    content: str


@dataclass(slots=True)
class Document:
    name: str
    text: str

COLLECTION = "kb_chunks"
EMB_SIZE = 1024
QDRANT_PATH = "./qdrant_chunks"

embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

#Использую формат pointstruct, не вижу смысла колонками, данных всего ничего
def chunks_to_points(chunks: list[LCDocument]) -> list[PointStruct]:

    points: list[PointStruct] = []
    for doc in chunks:
        vector = embedder.embed_query(doc.page_content)

        payload = dict(doc.metadata)
        payload["content"] = doc.page_content

        point_id = hashlib.md5(
            f"{payload.get('source', '')}::{doc.page_content}".encode("utf-8")
        ).hexdigest()

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    return points


def save_to_qdrant(
    points: list[PointStruct], batch_size: int = 64, rebuild: bool = False, ) -> int:
    client = QdrantClient(path=QDRANT_PATH)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing and rebuild:
        print(f"Удаляем коллекцию {COLLECTION}", flush=True)
        client.delete_collection(collection_name=COLLECTION)
        existing.remove(COLLECTION)

    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMB_SIZE, distance=Distance.COSINE),
        )

        from qdrant_client.models import PayloadSchemaType
        for field in ("source", "doc_type", "status", "doc_id"):
            client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    total = len(points)
    for i in range(0, total, batch_size):
        client.upsert(
            collection_name=COLLECTION,
            points=points[i : i + batch_size],
        )
        print(f"  загружено {min(i + batch_size, total)}/{total}", flush=True)

    count = client.count(collection_name=COLLECTION).count
    client.close()
    return count

def view_qdrant(
    limit: int = 5,
    show_vector: bool = False,
    preview_chars: int = 200,
) -> None:
    client = QdrantClient(path=QDRANT_PATH)

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
    client.close()

    for i, p in enumerate(points, 1):
        print(f"--- [{i}] id={p.id} ---")
        if show_vector and isinstance(p.vector, list):
            print(f"vector: len={len(p.vector)}, первые 5={p.vector[:5]}")

        payload = p.payload or {}
        meta = {k: v for k, v in payload.items() if k != "content"}
        content = payload.get("content", "")

        print(f"metadata: {meta}")
        print(f"content[:{preview_chars}]: {content[:preview_chars]}...")
        print()

def load_knowledge_base(zip_path: Path) -> list[Document]:
    result: list[Document] = []
    skipped = 0
    with zipfile.ZipFile(zip_path) as archive:
        names = [
            n for n in sorted(archive.namelist())
            if not n.endswith("/")
            and not n.startswith(("__MACOSX/", "."))
            and n.lower().endswith(".txt")
        ]
        for name in names:
            try:
                text = archive.read(name).decode("utf-8-sig")
            except (UnicodeDecodeError, KeyError):
                skipped += 1
                continue
            if text.strip():
                result.append(Document(name=name, text=text.strip()))
    return result

type_rules = [
    (["договор аренды"],                              "договор аренды"),
    (["договор поставки"],                            "договор поставки"),
    (["доп соглашение", "дополнительное соглашение"], "доп. соглашение"),
    (["коап", "фз", "кодекс", "статья"],              "нормативный акт"),
    (["комплаенс"],                                   "письмо"),
    (["регламент"],                                   "регламент"),
    (["служебная записка"],                           "служебная записка"),
]


def classify(stem: str, head: str) -> str:
    hay = (stem + " " + head[:300]).lower()
    for kws, doc_type in type_rules:
        if any(k in hay for k in kws):
            return doc_type
    return "прочее"



def extract_meta(path: Path, text: str) -> dict:
    stem = path.stem.split("__")[-1]
    head = text[:800]

    meta = {
        "source": stem,
        "doc_type": classify(stem, head),
    }
    number_doc = re.search(r"\s*([А-ЯA-Z]{1,4}-\d{2,4}/\d{4})", head)
    if number_doc:
        meta["doc_id"] = number_doc.group(0)
    
    if meta["doc_type"] == "регламент":
        low = text[:400].lower()
        meta["status"] = "утратил силу" if "утратил силу" in low else "действующий"

        m = re.search(r"Редакция\s+от\s+(\d{2}\.\d{2}\.\d{4})", head)
        meta["revision"] = m.group(1) if m else None

        m = re.search(r"Действует\s+с\s+(\d{2}\.\d{2}\.\d{4})", head)
        meta["effective_from"] = m.group(1) if m else None

        m = re.search(r"утрачивает силу(?:\s+с)?\s+(\d{2}\.\d{2}\.\d{4})", head, re.I)
        meta["effective_to"] = m.group(1) if m else None
    
    return meta


_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n",
        r"\n(?=Статья\s+\d+)",
        r"\n(?=Часть\s+\d+)",
        r"\n(?=\d+\.\s)",
        r"\n(?=\d+\.\d+\.)",
        "\n", ". ", " ", "",],
    keep_separator=True,
    is_separator_regex=True,
)


def split_document(doc: Document) -> list[LCDocument]:
    lc_doc = LCDocument(page_content=doc.text, metadata=extract_meta(Path(doc.name), doc.text))
    chunks = _splitter.split_documents([lc_doc])
    
    return chunks

kb = load_knowledge_base(Path("knowledge-base.zip"))
chunks = list(chain.from_iterable(split_document(doc) for doc in kb))

print("=== 1. Преобразование в векторы ===", flush=True)
points = chunks_to_points(chunks)
print(f"Готово: {len(points)} PointStruct, dim={len(points[0].vector)}\n", flush=True)

print("=== 2. Сохранение в Qdrant ===", flush=True)
count = save_to_qdrant(points, rebuild=True)
print(f"В коллекции: {count} точек\n", flush=True)

print("=== 3. Просмотр из Qdrant ===", flush=True)
view_qdrant(limit=5, show_vector=True)