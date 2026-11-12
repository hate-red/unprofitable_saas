from pathlib import Path
import re

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from schemas import Document

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

def split_documents(knowledge_base: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
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

    docs = []
    for doc in knowledge_base:
        lc_doc = LCDocument(page_content=doc.text, metadata=extract_meta(Path(doc.name), doc.text))
        docs.append(lc_doc)

    chunks = splitter.split_documents(docs)
    return chunks
