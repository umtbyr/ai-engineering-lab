"""Application-owned models: parser-library details do not leak past this layer."""

from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["heading", "paragraph", "code", "list_item"]


@dataclass(frozen=True)
class DocumentBlock:
    kind: BlockKind
    content: str
    heading_path: tuple[str, ...]
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    language: str | None = None
    heading_level: int | None = None


@dataclass(frozen=True)
class Section:
    id: str
    parent_section_id: str | None
    heading_path: tuple[str, ...]
    blocks: tuple[DocumentBlock, ...]


@dataclass(frozen=True)
class StructuredDocument:
    document_id: str
    source: str
    title: str
    blocks: tuple[DocumentBlock, ...]
    schema_version: int = 1


@dataclass(frozen=True)
class Chunk:
    strategy: str
    text: str
    heading_path: tuple[str, ...]
    source_lines: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    id: str | None = None
    section_id: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
