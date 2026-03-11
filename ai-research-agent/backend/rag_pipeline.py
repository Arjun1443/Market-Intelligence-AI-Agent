from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT_DIR = Path(__file__).resolve().parents[1]
for import_path in (ROOT_DIR / "backend", ROOT_DIR / "config"):
    if str(import_path) not in sys.path:
        sys.path.append(str(import_path))

from agents import AnalysisAgent, ReportAgent, SearchAgent  # noqa: E402
from embeddings import EmbeddingProvider, QdrantResearchStore  # noqa: E402
from settings import Settings  # noqa: E402


class ResearchPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding_provider = EmbeddingProvider(settings)
        self.vector_store = QdrantResearchStore(settings, self.embedding_provider)
        self.search_agent = SearchAgent(settings)
        self.analysis_agent = AnalysisAgent(settings, self.embedding_provider)
        self.report_agent = ReportAgent(self.embedding_provider)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def run_research(self, query: str, max_results: int = 10) -> dict[str, object]:
        sources = self.search_agent.run(query=query, max_results=max_results)
        documents = self._source_documents(sources)
        chunks = self.text_splitter.split_documents(documents)
        collection_name = self.vector_store.collection_name_for_query(query)
        indexed_ids = self.vector_store.index_documents(collection_name=collection_name, documents=chunks)
        analysis = self.analysis_agent.analyze(query=query, sources=sources)

        return {
            "query": query,
            "collection_name": collection_name,
            "source_count": len(sources),
            "chunk_count": len(chunks),
            "indexed_ids": indexed_ids,
            "sources": sources,
            "insights": analysis,
            "mode": {
                "apify": "live" if self.settings.apify_api_token else "demo",
                "llm": "openai" if self.settings.openai_api_key else "heuristic",
                "vector_store": "qdrant",
            },
        }

    def generate_report(
        self,
        query: str,
        collection_name: str | None = None,
        top_k: int = 8,
    ) -> dict[str, object]:
        target_collection = collection_name or self.vector_store.collection_name_for_query(query)
        documents = self.vector_store.similarity_search(
            collection_name=target_collection,
            query=query,
            top_k=top_k,
        )
        if not documents:
            raise ValueError(
                "No indexed research documents found for this query. Run /research first to populate the vector store."
            )

        sources = self._documents_to_sources(documents)
        analysis = self.analysis_agent.analyze(query=query, sources=sources)
        report = self.report_agent.generate(query=query, analysis=analysis, retrieved_docs=documents)

        return {
            "query": query,
            "collection_name": target_collection,
            "retrieved_documents": len(documents),
            "sources": sources,
            "insights": analysis,
            "report": report,
        }

    def _source_documents(self, sources: list[dict[str, object]]) -> list[Document]:
        documents = []
        for source in sources:
            content = "\n".join(
                [
                    str(source.get("title", "")),
                    str(source.get("snippet", "")),
                    str(source.get("url", "")),
                ]
            ).strip()
            documents.append(Document(page_content=content, metadata=source))
        return documents

    def _documents_to_sources(self, documents: list[Document]) -> list[dict[str, object]]:
        sources: list[dict[str, object]] = []
        seen_urls: set[str] = set()
        for doc in documents:
            metadata = dict(doc.metadata or {})
            url = str(metadata.get("url", ""))
            if url in seen_urls:
                continue
            seen_urls.add(url)
            metadata.setdefault("snippet", doc.page_content[:300].replace("\n", " "))
            metadata.setdefault("title", "Retrieved source")
            metadata.setdefault("source_type", "retrieved")
            sources.append(metadata)
        return sources
