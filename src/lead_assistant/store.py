"""ChromaDB vector store and embeddings, shared by ingest and query code."""

from __future__ import annotations

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from lead_assistant.config import Settings


class IndexMismatchError(RuntimeError):
    """The Chroma index was built with a different embedding model than the one configured."""


def get_embeddings(settings: Settings) -> Embeddings:
    model = settings.embedding_model
    if model.startswith("models/gemini"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if settings.gemini_api_key is None:
            raise RuntimeError(f"{model} needs the GEMINI_API_KEY environment variable")
        # The library sets task_type for us: RETRIEVAL_DOCUMENT for chunks,
        # RETRIEVAL_QUERY for questions.
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=settings.gemini_api_key.get_secret_value())

    from langchain_huggingface import HuggingFaceEmbeddings

    # Local sentence-transformers model, downloaded once into the Hugging Face cache.
    # normalize_embeddings=True makes vectors unit length, so cosine similarity is a
    # plain dot product. The query prefix applies to questions only, not documents:
    # bge is trained to match "instruction + short query" against plain passages.
    query_kwargs = {"normalize_embeddings": True}
    if settings.embedding_query_prefix and "bge" in model.lower():
        query_kwargs["prompt"] = settings.embedding_query_prefix
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs=query_kwargs,
    )


def _client(settings: Settings) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def get_vectorstore(settings: Settings) -> Chroma:
    client = _client(settings)
    existing = {c.name for c in client.list_collections()}
    if settings.collection_name in existing:
        # Vectors from different models live in unrelated spaces (here even different
        # sizes), so querying an index with the wrong model returns nonsense or crashes.
        built_with = (client.get_collection(settings.collection_name).metadata or {}).get("embedding_model")
        if built_with != settings.embedding_model:
            raise IndexMismatchError(
                f"Index was built with {built_with or 'an unrecorded model'}, but EMBEDDING_MODEL is "
                f"{settings.embedding_model}. Re-run: python -m lead_assistant.ingest")
    return Chroma(
        client=client,
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(settings),
        # Cosine distance, so relevance = 1 - distance is comparable across queries.
        # embedding_model is recorded so a later model change is caught (see above).
        collection_metadata={"hnsw:space": "cosine", "embedding_model": settings.embedding_model},
    )


def reset_collection(settings: Settings) -> None:
    client = _client(settings)
    if settings.collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(settings.collection_name)
