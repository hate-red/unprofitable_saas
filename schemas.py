from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Question:
    id: str
    content: str


@dataclass(slots=True)
class Document:
    path: Path
    content: str
    metadata: dict

    def to_dict(self) -> dict:
        return  {
            'path': self.path,
            'content': self.content,
            'metadata': self.metadata
        }
