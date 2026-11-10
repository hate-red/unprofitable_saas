import sys
import csv
import io
import zipfile

from pathlib import Path
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import config


def words(text: str) -> set[str]:
    return set(config.WORD_RE.findall(text.lower().replace("ё", "е")))


def snippet(text: str) -> str:
    piece = text[:config.SNIPPET_CHARS]
    
    if config.CHUNKER == "common" and len(text) > config.SNIPPET_CHARS:
        boundary = piece.rfind("\n")
        if boundary > config.SNIPPET_CHARS // 2:
            piece = piece[:boundary]
    
    return piece.strip()


def find_file(*names: str) -> Path:
    for root in (config.DATA_DIR, Path(__file__).resolve().parent / config.DATA_DIR, Path.cwd()):
        for name in names:
            path = root / name
            if path.is_file():
                return path
    
    raise FileNotFoundError(f"Не найден один из файлов: {', '.join(names)}")


def load_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("В questions.csv нужны колонки id и question")
    
        rows = [(str(r["id"]).strip(), str(r["question"]).strip()) for r in reader]
    
    if not rows or any(not qid for qid, _ in rows):
        raise ValueError("В questions.csv отсутствуют вопросы или id")
    
    return rows


def read_docx(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as doc:
        xml = doc.read("word/document.xml")
    
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    
    return "\n".join(
        "".join(x.text or "" for x in paragraph.iter(ns + "t"))
        for paragraph in root.iter(ns + "p")
    )


def load_documents(path: Path) -> list[tuple[str, str]]:
    documents = []

    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith("/") or name.startswith("__MACOSX/"):
                continue
            
            suffix = Path(name).suffix.lower()
            if suffix not in {".txt", ".md", ".docx"}:
                continue
            
            raw = archive.read(name)
            try:
                text = read_docx(raw) if suffix == ".docx" else raw.decode("utf-8-sig")
            except (UnicodeDecodeError, KeyError, ValueError, zipfile.BadZipFile, ET.ParseError):
                print(f"Пропущен нечитаемый документ: {name}", file=sys.stderr)
                continue
            
            if text.strip():
                documents.append((name, text.strip()))
    
    if not documents:
        raise ValueError("В архиве базы знаний нет читаемых документов")
    
    return documents


def run_workers(questions: list[tuple[str, str]], work):
    answers = [config.FALLBACK] * len(questions)
    
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [pool.submit(work, i, question) for i, (_, question) in enumerate(questions)]
    
        for finished, future in enumerate(as_completed(futures), 1):
            index, answer = future.result()
            answers[index] = answer

            if finished % 10 == 0 or finished == len(questions):
                print(f"Обработано {finished}/{len(questions)}", flush=True)

    return answers


def write_submission_file(
    questions: list[tuple[str, str]], 
    answers: list[str],
) -> None:
    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
    
        for i, (qid, _) in enumerate(questions):
            writer.writerow([qid, answers[i]])
    
    print(f"Создан submission.csv: {len(questions)} ответов", flush=True)
