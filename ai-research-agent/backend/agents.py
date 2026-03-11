from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sklearn.cluster import KMeans

ROOT_DIR = Path(__file__).resolve().parents[1]
for import_path in (ROOT_DIR / "backend", ROOT_DIR / "config"):
    if str(import_path) not in sys.path:
        sys.path.append(str(import_path))

from embeddings import EmbeddingProvider  # noqa: E402
from scraper import ApifyWebScraper  # noqa: E402
from settings import Settings  # noqa: E402


class SearchAgent:
    def __init__(self, settings: Settings) -> None:
        self.scraper = ApifyWebScraper(settings)

    def run(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        return self.scraper.collect(query=query, max_results=max_results)


class AnalysisAgent:
    def __init__(self, settings: Settings, embedding_provider: EmbeddingProvider) -> None:
        self.settings = settings
        self.embedding_provider = embedding_provider

    def analyze(self, query: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
        if not sources:
            raise ValueError("No sources were collected for analysis.")

        texts = [self._combine_text(source) for source in sources]
        entities = self._extract_entities(sources)
        trends = self._extract_trends(texts)
        clusters = self._cluster_sources(sources, texts)
        insight = self._market_insight(query, sources, entities, trends, clusters)
        opportunities = self._future_opportunities(trends)

        return {
            "entities": entities,
            "trends": trends,
            "clusters": clusters,
            "market_insight": insight,
            "future_opportunities": opportunities,
        }

    def _combine_text(self, source: dict[str, Any]) -> str:
        title = source.get("title", "")
        snippet = source.get("snippet", "")
        url = source.get("url", "")
        return f"{title}\n{snippet}\n{url}".strip()

    def _extract_entities(self, sources: list[dict[str, Any]]) -> list[str]:
        stop_terms = {
            "The",
            "This",
            "That",
            "These",
            "Those",
            "Healthcare",
            "Health",
            "Market",
            "Emerging",
            "Future",
            "Startup",
            "Startups",
            "Report",
        }
        candidates: Counter[str] = Counter()

        for source in sources:
            title = source.get("title", "")
            title_head = re.split(r"[-|:]", title)[0].strip()
            if 1 <= len(title_head.split()) <= 5:
                candidates[title_head] += 2

            joined = f"{title} {source.get('snippet', '')}"
            matches = re.findall(
                r"\b(?:[A-Z][a-zA-Z0-9&.+-]+(?:\s+[A-Z][a-zA-Z0-9&.+-]+){0,3}|AI)\b",
                joined,
            )
            for match in matches:
                cleaned = match.strip()
                if cleaned in stop_terms or len(cleaned) < 2:
                    continue
                candidates[cleaned] += 1

        entities = [item for item, _ in candidates.most_common(8)]
        return entities or ["No dominant entities identified"]

    def _extract_trends(self, texts: list[str]) -> list[str]:
        tokens: Counter[str] = Counter()
        for text in texts:
            phrases = re.findall(r"\b[a-zA-Z][a-zA-Z-]{3,}\b", text.lower())
            for phrase in phrases:
                if phrase in {
                    "about",
                    "their",
                    "there",
                    "which",
                    "using",
                    "market",
                    "startup",
                    "startups",
                    "analysis",
                }:
                    continue
                tokens[phrase] += 1

        trends = [token.replace("-", " ") for token, count in tokens.most_common(10) if count > 1]
        return trends[:6] or ["Limited trend signal available from collected sources"]

    def _cluster_sources(self, sources: list[dict[str, Any]], texts: list[str]) -> list[dict[str, Any]]:
        if len(texts) == 1:
            source = sources[0]
            return [
                {
                    "theme": source.get("title", "Single source cluster"),
                    "document_count": 1,
                    "examples": [source.get("title", "Untitled source")],
                }
            ]

        vectors = np.array(self.embedding_provider.embed_documents(texts))
        cluster_count = min(3, len(texts))
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        labels = model.fit_predict(vectors)

        grouped: dict[int, list[dict[str, Any]]] = {}
        for label, source in zip(labels, sources):
            grouped.setdefault(int(label), []).append(source)

        clusters: list[dict[str, Any]] = []
        for cluster_sources in grouped.values():
            words: Counter[str] = Counter()
            titles = [item.get("title", "Untitled source") for item in cluster_sources]
            for source in cluster_sources:
                text = f"{source.get('title', '')} {source.get('snippet', '')}".lower()
                for token in re.findall(r"\b[a-zA-Z][a-zA-Z-]{3,}\b", text):
                    if token in {"with", "from", "that", "this", "have", "healthcare"}:
                        continue
                    words[token] += 1
            keywords = [word.replace("-", " ") for word, _ in words.most_common(3)]
            theme = ", ".join(keywords).title() if keywords else "Related market theme"
            clusters.append(
                {
                    "theme": theme,
                    "document_count": len(cluster_sources),
                    "examples": titles[:3],
                }
            )

        clusters.sort(key=lambda item: item["document_count"], reverse=True)
        return clusters

    def _market_insight(
        self,
        query: str,
        sources: list[dict[str, Any]],
        entities: list[str],
        trends: list[str],
        clusters: list[dict[str, Any]],
    ) -> str:
        llm = self.embedding_provider.get_chat_model()
        if llm is None:
            source_types = Counter(source.get("source_type", "web") for source in sources)
            dominant_type = source_types.most_common(1)[0][0]
            leading_theme = clusters[0]["theme"] if clusters else "broad market coverage"
            return (
                f"Collected sources suggest the market is coalescing around {leading_theme.lower()}, "
                f"with recurring attention on {', '.join(trends[:3])}. Coverage is currently led by "
                f"{dominant_type} sources, which suggests the topic is still being shaped by rapid external "
                f"signals. Notable organizations appearing most often include {', '.join(entities[:4])}."
            )

        prompt = ChatPromptTemplate.from_template(
            """
            You are a market intelligence analyst.
            Research question: {query}

            Entities: {entities}
            Trends: {trends}
            Clusters: {clusters}
            Source titles: {titles}

            Write one concise paragraph with market insights. Focus on momentum, competitive pattern,
            and why this matters now.
            """
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke(
            {
                "query": query,
                "entities": ", ".join(entities[:6]),
                "trends": ", ".join(trends[:6]),
                "clusters": json.dumps(clusters[:3]),
                "titles": json.dumps([source.get("title", "") for source in sources[:8]]),
            }
        ).strip()

    def _future_opportunities(self, trends: list[str]) -> list[str]:
        seeds = trends[:3] or ["specialized AI workflows", "data infrastructure", "workflow automation"]
        return [
            f"Build differentiated products around {seed} with a narrower workflow or customer segment focus."
            for seed in seeds
        ]


class ReportAgent:
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider

    def generate(self, query: str, analysis: dict[str, Any], retrieved_docs: list[Document]) -> str:
        llm = self.embedding_provider.get_chat_model()
        context = self._build_context(retrieved_docs)
        if llm is None:
            return self._build_fallback_report(analysis, retrieved_docs)

        prompt = ChatPromptTemplate.from_template(
            """
            You are generating a structured market research report.

            Research question: {query}
            Analysis summary: {analysis}
            Retrieved context:
            {context}

            Produce Markdown using exactly this format:
            # Research Report

            ## Key Companies
            List companies with short descriptions

            ## Emerging Trends
            Bullet list of trends

            ## Market Insights
            One or two paragraphs

            ## Future Opportunities
            Bullet list of opportunities
            """
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke(
            {
                "query": query,
                "analysis": json.dumps(analysis),
                "context": context,
            }
        ).strip()

    def _build_context(self, documents: list[Document]) -> str:
        sections = []
        for index, doc in enumerate(documents, start=1):
            metadata = doc.metadata or {}
            sections.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"Title: {metadata.get('title', 'Untitled')}",
                        f"URL: {metadata.get('url', 'N/A')}",
                        f"Content: {doc.page_content[:1200]}",
                    ]
                )
            )
        return "\n\n".join(sections)

    def _build_fallback_report(self, analysis: dict[str, Any], retrieved_docs: list[Document]) -> str:
        companies = []
        seen_titles: set[str] = set()
        for doc in retrieved_docs:
            metadata = doc.metadata or {}
            title = metadata.get("title", "Untitled source")
            if title in seen_titles:
                continue
            seen_titles.add(title)
            snippet = metadata.get("snippet") or doc.page_content[:180].replace("\n", " ")
            companies.append(f"- **{title}**: {snippet}")
            if len(companies) == 5:
                break

        trends = analysis.get("trends", [])
        opportunities = analysis.get("future_opportunities", [])
        insight = analysis.get("market_insight", "No market insight generated.")

        trend_lines = "\n".join(f"- {trend}" for trend in trends[:6]) or "- No trends identified"
        company_lines = "\n".join(companies) or "- No company-level signals identified"
        opportunity_lines = "\n".join(f"- {item}" for item in opportunities[:5]) or "- No opportunities identified"

        return f"""# Research Report

## Key Companies
{company_lines}

## Emerging Trends
{trend_lines}

## Market Insights
{insight}

## Future Opportunities
{opportunity_lines}
"""
