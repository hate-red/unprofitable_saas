from dataclasses import dataclass


@dataclass(slots=True)
class Question:
    id: str
    content: str


@dataclass(slots=True)
class Document:
    name: str
    text: str
