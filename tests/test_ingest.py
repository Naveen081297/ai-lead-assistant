"""Offline tests for loading and chunking (no API calls)."""

from pathlib import Path

from lead_assistant.ingest import load_documents, split_documents
from lead_assistant.rag import Passage, RagPipeline

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


def test_loads_every_markdown_doc_with_title():
    docs = load_documents(DOCS)
    assert len(docs) == len(list(DOCS.glob("*.md")))
    fee = next(d for d in docs if d.metadata["source"] == "fee-sheet-2027.md")
    assert fee.metadata["title"] == "Fee Sheet 2027"


def test_chunks_are_bounded_unique_and_titled():
    chunks = split_documents(load_documents(DOCS), chunk_size=1000, chunk_overlap=150)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique (used as Chroma ids)"
    for c in chunks:
        title = c.metadata["title"]
        assert c.page_content.startswith((f"[{title}]", f"# {title}"))
        # chunk_size limit plus the short title prefix we add
        assert len(c.page_content) <= 1000 + len(title) + 3


def test_fee_table_stays_in_one_chunk():
    chunks = split_documents(load_documents(DOCS), chunk_size=1000, chunk_overlap=150)
    hits = [c for c in chunks if "1,05,020" in c.page_content]
    assert hits and all("Data Analytics Pro" in c.page_content for c in hits)


def test_context_numbers_passages_with_sources():
    p = Passage(n=2, chunk_id="a.md#0-0", source="a.md", title="A", page=None, score=0.8, text="hello")
    assert RagPipeline.build_context([p]) == "[2] (source: a.md)\nhello"
