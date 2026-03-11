from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from apify_client import ApifyClient

ROOT_DIR = Path(__file__).resolve().parents[1]
for import_path in (ROOT_DIR / "backend", ROOT_DIR / "config"):
    if str(import_path) not in sys.path:
        sys.path.append(str(import_path))

from settings import Settings  # noqa: E402


class ApifyWebScraper:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = ApifyClient(settings.apify_api_token) if settings.apify_api_token else None

    def collect(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if self.client is None:
            return self._demo_results(query=query, max_results=max_results)

        actor_client = self.client.actor(self.settings.apify_actor_id)
        run_input = {
            "queries": [query],
            "maxPagesPerQuery": max(1, min(3, max_results // 10 + 1)),
            "resultsPerPage": min(max_results, 10),
            "countryCode": self.settings.apify_country_code,
            "languageCode": self.settings.apify_language_code,
            "mobileResults": False,
        }
        run = actor_client.call(run_input=run_input)
        dataset_id = self._extract_dataset_id(run)
        items_result = self.client.dataset(dataset_id).list_items()
        items = getattr(items_result, "items", None) or items_result.get("items", [])

        parsed_results = self._parse_apify_items(items, max_results=max_results)
        if not parsed_results:
            raise ValueError("Apify returned no usable search results for this query.")
        return parsed_results

    def _extract_dataset_id(self, run: Any) -> str:
        if run is None:
            raise ValueError("Apify actor run failed.")

        dataset_id = getattr(run, "default_dataset_id", None)
        if dataset_id:
            return dataset_id

        if isinstance(run, dict):
            dataset_id = run.get("defaultDatasetId") or run.get("default_dataset_id")
            if dataset_id:
                return dataset_id

        raise ValueError("Apify actor run did not include a dataset identifier.")

    def _parse_apify_items(self, items: list[dict[str, Any]], max_results: int) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []

        for item in items:
            organic_results = item.get("organicResults")
            if isinstance(organic_results, list):
                for organic in organic_results:
                    parsed = self._normalize_result(organic)
                    if parsed:
                        sources.append(parsed)
                continue

            parsed = self._normalize_result(item)
            if parsed:
                sources.append(parsed)

        deduped: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for source in sources:
            url = source.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append(source)
            if len(deduped) >= max_results:
                break

        return deduped

    def _normalize_result(self, item: dict[str, Any]) -> dict[str, Any] | None:
        url = item.get("url") or item.get("link")
        title = item.get("title") or item.get("name")
        snippet = (
            item.get("description")
            or item.get("snippet")
            or item.get("text")
            or item.get("markdown")
            or ""
        )
        if not url and not title and not snippet:
            return None

        source_type = self._classify_source(url or "", title or "", snippet)
        clean_snippet = self._clean_text(snippet)
        domain = urlparse(url).netloc.replace("www.", "") if url else "unknown"

        return {
            "title": title or domain or "Untitled source",
            "url": url or "",
            "snippet": clean_snippet[:1200],
            "domain": domain,
            "source_type": source_type,
        }

    def _classify_source(self, url: str, title: str, snippet: str) -> str:
        text = f"{url} {title} {snippet}".lower()
        if "news" in text or any(domain in text for domain in {"techcrunch", "reuters", "forbes"}):
            return "news"
        if "blog" in text or "medium.com" in text or "substack" in text:
            return "blog"
        if "startup" in text or "company" in text:
            return "startup"
        return "web"

    def _clean_text(self, text: str) -> str:
        import re

        text = re.sub(r"\s+", " ", text or "")
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        return text.strip()

    def _demo_results(self, query: str, max_results: int) -> list[dict[str, Any]]:
        base = [
            {
                "title": f"{query} market overview",
                "url": "https://demo.local/market-overview",
                "snippet": (
                    f"Demo mode result summarizing the market landscape for {query}. "
                    "Configure APIFY_API_TOKEN to replace demo content with live web data."
                ),
                "domain": "demo.local",
                "source_type": "demo",
            },
            {
                "title": f"{query} startup activity",
                "url": "https://demo.local/startup-activity",
                "snippet": (
                    f"Demo mode result describing startup activity, funding patterns, and product themes around {query}."
                ),
                "domain": "demo.local",
                "source_type": "demo",
            },
            {
                "title": f"{query} emerging trends",
                "url": "https://demo.local/emerging-trends",
                "snippet": (
                    f"Demo mode result capturing technology shifts, regulation, and adoption signals relevant to {query}."
                ),
                "domain": "demo.local",
                "source_type": "demo",
            },
            {
                "title": f"{query} enterprise adoption",
                "url": "https://demo.local/enterprise-adoption",
                "snippet": (
                    f"Demo mode result focused on enterprise use cases, deployment models, and market readiness in {query}."
                ),
                "domain": "demo.local",
                "source_type": "demo",
            },
            {
                "title": f"{query} future opportunities",
                "url": "https://demo.local/future-opportunities",
                "snippet": (
                    f"Demo mode result outlining white-space opportunities, product gaps, and next-wave innovation areas for {query}."
                ),
                "domain": "demo.local",
                "source_type": "demo",
            },
        ]
        return base[:max_results]
