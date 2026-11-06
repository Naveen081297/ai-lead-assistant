"""Load course documents, split them into chunks, embed them and store them in ChromaDB.

Run:  python -m lead_assistant.ingest
"""

from __future__ import annotations

import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lead_assistant.config import Settings, get_settings
from lead_assistant.store import get_vectorstore, reset_collection

SUPPORTED = {".md", ".txt", ".pdf"}


# ---------- 1. Load ----------

def _title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_documents(docs_dir: Path) -> list[Document]:
    """One Document per text file, one per page for PDFs (so citations can name the page)."""
    docs: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED:
            continue
        source = path.relative_to(docs_dir).as_posix()
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(path)
            for page_no, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": source, "title": path.stem, "page": page_no},
                    ))
        else:
            text = path.read_text(encoding="utf-8")
            docs.append(Document(
                page_content=text,
                metadata={"source": source, "title": _title_from_markdown(text, path.stem)},
            ))
    return docs


# ---------- 2. Split ----------

def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split into overlapping chunks.

    RecursiveCharacterTextSplitter tries the separators in order: markdown headings
    first, then blank lines (paragraphs), then single lines, then sentences, then
    words. So it only cuts mid-paragraph when a paragraph is bigger than chunk_size.

    Each chunk gets the document title prepended. A chunk like "Instalments: 3 x
    Rs 30,000" means nothing on its own; "Data Analytics Program > Instalments..."
    is findable. This is a cheap version of "contextual chunk headers".
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    per_source: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata["source"]
        idx = per_source.get(src, 0)
        per_source[src] = idx + 1
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = f"{src}#{chunk.metadata.get('page', 0)}-{idx}"
        title = chunk.metadata["title"]
        if not chunk.page_content.lstrip().startswith(f"# {title}"):
            chunk.page_content = f"[{title}]\n{chunk.page_content}"
    return chunks


# ---------- 3. Embed + store ----------

def ingest(settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    docs = load_documents(settings.docs_dir)
    if not docs:
        raise SystemExit(f"No documents found in {settings.docs_dir}")
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)

    # Full rebuild each time: simple and keeps the index in sync with the folder
    # (deleted or edited files don't leave stale chunks behind).
    reset_collection(settings)
    store = get_vectorstore(settings)
    # Chroma calls the embedding model (task_type=RETRIEVAL_DOCUMENT) for each chunk.
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return len(chunks)


def main() -> None:
    settings = get_settings()
    t0 = time.perf_counter()
    docs = load_documents(settings.docs_dir)
    n = ingest(settings)
    print(f"Indexed {len(docs)} documents as {n} chunks into "
          f"'{settings.collection_name}' at {settings.chroma_dir} "
          f"in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
