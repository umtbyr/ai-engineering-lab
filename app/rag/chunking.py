"""Two deliberately simple policies; character limits are not token limits."""

from app.rag.models import Chunk, DocumentBlock, StructuredDocument, Section


def _build(blocks: list[DocumentBlock], strategy: str) -> Chunk:
    return Chunk(
        strategy=strategy,
        text="\n\n".join(b.content for b in blocks),
        heading_path=blocks[-1].heading_path,
        source_lines=tuple((b.start_line, b.end_line) for b in blocks),
    )


def chunk_by_section(document: StructuredDocument) -> list[Chunk]:
    """Keep consecutive blocks belonging to the same heading path together."""
    result: list[Chunk] = []
    group: list[DocumentBlock] = []
    for block in document.blocks:
        if group and block.heading_path != group[-1].heading_path:
            result.append(_build(group, "section"))
            group = []
        group.append(block)
    if group:
        result.append(_build(group, "section"))
    return result


def chunk_by_size(document: StructuredDocument, max_chars: int = 250) -> list[Chunk]:
    """Pack whole blocks up to a character budget; never split a code block."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    result: list[Chunk] = []
    group: list[DocumentBlock] = []
    size = 0
    for block in document.blocks:
        additional = len(block.content) + (2 if group else 0)
        if group and size + additional > max_chars:
            result.append(_build(group, "size"))
            group, size = [], 0
        group.append(block)
        size += len(block.content) + (2 if len(group) > 1 else 0)
    if group:
        result.append(_build(group, "size"))
    return result


def build_sections(document: StructuredDocument) -> list[Section]:

    # ADIM 1: DocumentBlock'ları gruplandır

    groups: list[list[DocumentBlock]] = []
    current_group: list[DocumentBlock] = []

    for block in document.blocks:

        is_new_section = block.kind == "heading" or (
            current_group and block.heading_path != current_group[-1].heading_path
        )

        if current_group and is_new_section:
            groups.append(current_group)
            current_group = []

        current_group.append(block)

    if current_group:
        groups.append(current_group)

    # ADIM 2: Gruplardan Section nesneleri oluştur

    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []

    for index, blocks in enumerate(groups):

        first_block = blocks[0]
        section_id = f"{document.document_id}:s{index}"

        parent_id = None

        if first_block.kind == "heading":

            level = first_block.heading_level

            if level is None:
                raise ValueError("Heading level is required")

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            if heading_stack:
                parent_id = heading_stack[-1][1]

            heading_stack.append((level, section_id))

        elif heading_stack:
            parent_id = heading_stack[-1][1]

        sections.append(
            Section(
                id=section_id,
                parent_section_id=parent_id,
                heading_path=first_block.heading_path,
                blocks=tuple(blocks),
            )
        )

    return sections


def chunk_sections(
    sections: list[Section],
    max_chars: int = 500,
) -> list[Chunk]:

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    chunks: list[Chunk] = []

    for section in sections:

        # 1. Heading bloklarını çıkar.
        content_blocks = [block for block in section.blocks if block.kind != "heading"]

        if not content_blocks:
            continue

        # 2. Chunk'a başlık bağlamını ekleyeceğiz.
        heading = " > ".join(section.heading_path)
        prefix = f"{heading}\n\n" if heading else ""

        # 3. Section içindeki blokları boyuta göre grupla.
        groups: list[list[DocumentBlock]] = []
        current_group: list[DocumentBlock] = []
        current_size = len(prefix)

        for block in content_blocks:

            separator_size = 2 if current_group else 0
            block_size = len(block.content) + separator_size

            if current_group and current_size + block_size > max_chars:
                groups.append(current_group)

                current_group = []
                current_size = len(prefix)
                block_size = len(block.content)

            current_group.append(block)
            current_size += block_size

        if current_group:
            groups.append(current_group)

        # 4. Gruplardan bağlantılı Chunk nesneleri oluştur.
        for index, group in enumerate(groups):

            chunk_id = f"{section.id}:c{index}"

            previous_id = None
            next_id = None

            if index > 0:
                previous_id = f"{section.id}:c{index - 1}"

            if index < len(groups) - 1:
                next_id = f"{section.id}:c{index + 1}"

            body = "\n\n".join(block.content for block in group)

            chunks.append(
                Chunk(
                    id=chunk_id,
                    section_id=section.id,
                    strategy="section_aware",
                    text=prefix + body,
                    heading_path=section.heading_path,
                    source_lines=tuple(
                        (block.start_line, block.end_line) for block in group
                    ),
                    previous_chunk_id=previous_id,
                    next_chunk_id=next_id,
                )
            )

    return chunks
