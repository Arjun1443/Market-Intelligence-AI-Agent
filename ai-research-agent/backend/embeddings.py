from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

ROOT_DIR = Path(__file__).resolve().parents[1]
for import_path in (ROOT_DIR / "backend", ROOT_DIR / "config"):
    if str(import_path) not in sys.path:
        sys.path.append(str(import_path))

from settings import Settings  # noqa: E402


class SimpleHashEmbeddings(Embeddings):
    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / magnitude for value in vector]


class EmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._embedding_model = self._build_embedding_model()
        self._chat_model = self._build_chat_model()

    def _build_embedding_model(self) -> Embeddings:
        if self.settings.openai_api_key:
            return OpenAIEmbeddings(
                api_key=self.settings.openai_api_key,
                model=self.settings.embedding_model,
            )
        return SimpleHashEmbeddings(dimension=self.settings.fallback_embedding_dimension)

    def _build_chat_model(self) -> ChatOpenAI | None:
        if not self.settings.openai_api_key:
            return None
        return ChatOpenAI(
            api_key=self.settings.openai_api_key,
            model=self.settings.chat_model,
            temperature=0.2,
        )

    def get_embedding_model(self) -> Embeddings:
        return self._embedding_model

    def get_chat_model(self) -> ChatOpenAI | None:
        return self._chat_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedding_model.embed_documents(texts)


class QdrantResearchStore:
    def __init__(self, settings: Settings, embedding_provider: EmbeddingProvider) -> None:
        self.settings = settings
        self.embedding_provider = embedding_provider
        self.client = self._build_client()

    def _build_client(self) -> QdrantClient:
        if self.settings.qdrant_url:
            return QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
            )

        data_dir = Path(self.settings.qdrant_local_path)
        data_dir.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(data_dir))

    def collection_name_for_query(self, query: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
        slug = slug[:40] or "research-query"
        suffix = hashlib.md5(query.encode("utf-8")).hexdigest()[:8]
        return f"market-intel-{slug}-{suffix}"

    def index_documents(self, collection_name: str, documents: list[Document]) -> list[str]:
        self._ensure_collection(collection_name)
        vector_store = self._vector_store(collection_name)
        ids = [hashlib.md5(f"{collection_name}-{idx}".encode("utf-8")).hexdigest() for idx, _ in enumerate(documents)]
        vector_store.add_documents(documents=documents, ids=ids)
        return ids

    def similarity_search(self, collection_name: str, query: str, top_k: int = 8) -> list[Document]:
        if not self.client.collection_exists(collection_name):
            return []
        vector_store = self._vector_store(collection_name)
        return vector_store.similarity_search(query=query, k=top_k)

    def _ensure_collection(self, collection_name: str) -> None:
        if self.client.collection_exists(collection_name):
            return

        probe_vector = self.embedding_provider.get_embedding_model().embed_query("market intelligence probe")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(probe_vector),
                distance=Distance.COSINE,
            ),
        )

    def _vector_store(self, collection_name: str) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embedding_provider.get_embedding_model(),
        )
