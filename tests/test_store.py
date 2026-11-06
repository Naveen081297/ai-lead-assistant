"""Offline tests for the vector-store guard (no model download, no API calls)."""

import chromadb
import pytest

from lead_assistant.config import Settings
from lead_assistant.store import IndexMismatchError, get_vectorstore


def settings(tmp_path, **overrides) -> Settings:
    return Settings(CHROMA_DIR=str(tmp_path), LEAD_ASSISTANT_CLAUDE_KEY="x", **overrides)


def make_index(tmp_path, metadata):
    chromadb.PersistentClient(path=str(tmp_path)).create_collection("course_docs", metadata=metadata)


def test_index_built_with_other_model_is_rejected(tmp_path):
    make_index(tmp_path, {"hnsw:space": "cosine", "embedding_model": "models/gemini-embedding-2"})
    with pytest.raises(IndexMismatchError, match="Re-run"):
        get_vectorstore(settings(tmp_path))


def test_index_without_recorded_model_is_rejected(tmp_path):
    make_index(tmp_path, {"hnsw:space": "cosine"})  # e.g. built before the guard existed
    with pytest.raises(IndexMismatchError, match="unrecorded"):
        get_vectorstore(settings(tmp_path))


def test_gemini_embeddings_without_key_fail_clearly(tmp_path):
    s = settings(tmp_path, EMBEDDING_MODEL="models/gemini-embedding-2", GEMINI_API_KEY=None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        get_vectorstore(s)


def test_default_embedding_model_is_local_bge():
    assert Settings(LEAD_ASSISTANT_CLAUDE_KEY="x").embedding_model == "BAAI/bge-small-en-v1.5"
