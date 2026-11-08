"""Convert Markdown parser tokens into our application's DocumentBlock models.

We do NOT parse Markdown syntax ourselves. ``markdown-it-py`` does that.
Our job is to translate its token stream into a stable, readable data model.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

from markdown_it import MarkdownIt
from markdown_it.token import Token

from app.rag.models import DocumentBlock, StructuredDocument

markdown_parser = MarkdownIt("commonmark")


@dataclass(frozen=True)
class Heading:
    """A heading currently in scope, e.g. H2 'Controller'."""

    level: int
    title: str


@dataclass(frozen=True)
class PendingBlock:
    """A heading or paragraph that has started but needs its inline text."""

    kind: Literal["heading", "paragraph", "list_item"]
    start_line: int
    end_line: int
    heading_level: int | None = None


def source_lines(token: Token) -> tuple[int, int]:
    """Convert markdown-it's [0-based start, exclusive end] to 1-based lines.

    If a token has no source map, we use (0, 0) to mean 'unknown'.
    """

    if not token.map:
        return (0, 0)

    start_line = token.map[0] + 1
    end_line = token.map[1]
    return (start_line, end_line)


def update_heading_stack(
    heading_stack: list[Heading],
    level: int,
    title: str,
) -> None:
    """Keep parent headings, drop siblings/children, then add the new heading.

    For example, H1 > H2 > H3 followed by a new H2 becomes H1 > new H2.
    """

    while heading_stack and heading_stack[-1].level >= level:
        heading_stack.pop()

    heading_stack.append(Heading(level=level, title=title))


def current_heading_path(heading_stack: list[Heading]) -> tuple[str, ...]:
    """Return a breadcrumb such as ('React Hook Form', 'Controller')."""

    return tuple(heading.title for heading in heading_stack)


def parse_markdown(content: str, *, source: str = "example.md") -> StructuredDocument:
    """Turn one Markdown string into a parser-independent StructuredDocument."""

    tokens = markdown_parser.parse(content)

    blocks: list[DocumentBlock] = []
    heading_stack: list[Heading] = []
    pending_block: PendingBlock | None = None
    list_depth = 0

    # Use the source filename unless a first H1 heading provides a better title.
    document_title = Path(source).stem
    found_h1 = False

    for token in tokens:
        # 1. Track whether paragraphs belong to a list item.
        if token.type in ("bullet_list_open", "ordered_list_open"):
            list_depth += 1

        elif token.type in ("bullet_list_close", "ordered_list_close"):
            list_depth -= 1

        # 2. An opening token describes the block, but not its inline text yet.
        elif token.type == "heading_open":
            heading_level = int(token.tag[1:])  # "h2" -> 2
            start_line, end_line = source_lines(token)

            pending_block = PendingBlock(
                kind="heading",
                start_line=start_line,
                end_line=end_line,
                heading_level=heading_level,
            )

        elif token.type == "paragraph_open":
            start_line, end_line = source_lines(token)
            block_kind = "list_item" if list_depth > 0 else "paragraph"

            pending_block = PendingBlock(
                kind=block_kind,
                start_line=start_line,
                end_line=end_line,
            )

        # 3. The inline token completes the previously opened block.
        elif token.type == "inline" and pending_block is not None:
            if pending_block.kind == "heading":
                level = pending_block.heading_level
                if level is None:
                    raise ValueError("Heading token is missing its level")

                update_heading_stack(heading_stack, level, token.content)

                if level == 1 and not found_h1:
                    document_title = token.content
                    found_h1 = True

            blocks.append(
                DocumentBlock(
                    kind=pending_block.kind,
                    content=token.content,
                    heading_path=current_heading_path(heading_stack),
                    start_line=pending_block.start_line,
                    end_line=pending_block.end_line,
                    heading_level=pending_block.heading_level,
                )
            )
            pending_block = None

        # 4. Fenced/indented code comes in one token; no pending state needed.
        elif token.type in ("fence", "code_block"):
            start_line, end_line = source_lines(token)
            language = None

            if token.info.strip():
                language = token.info.split()[0]

            blocks.append(
                DocumentBlock(
                    kind="code",
                    content=token.content.rstrip("\n"),
                    heading_path=current_heading_path(heading_stack),
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                )
            )

    # MVP scope: headings, paragraphs, simple list item paragraphs and code.
    # We do not preserve nested list structure or inline formatting as an AST.
    return StructuredDocument(
        document_id=sha256(content.encode("utf-8")).hexdigest()[:16],
        source=source,
        title=document_title,
        blocks=tuple(blocks),
    )
