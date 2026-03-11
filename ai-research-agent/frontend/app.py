from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Autonomous Market Intelligence AI Agent",
    page_icon=":mag:",
    layout="wide",
)

st.title("Autonomous Market Intelligence AI Agent")
st.caption("Research the web, build a Qdrant knowledge base, and generate structured market reports.")


def post_json(path: str, payload: dict):
    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{API_BASE_URL}{path}", json=payload)
        response.raise_for_status()
        return response.json()


def render_sources(sources: list[dict]) -> None:
    for index, source in enumerate(sources, start=1):
        with st.expander(f"{index}. {source.get('title', 'Untitled source')}"):
            st.write(source.get("snippet", ""))
            st.caption(f"Type: {source.get('source_type', 'web')}")
            if source.get("url"):
                st.markdown(f"[Open source]({source['url']})")


def render_insights(insights: dict) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Entities")
        st.write(insights.get("entities", []))
        st.subheader("Emerging Trends")
        for trend in insights.get("trends", []):
            st.markdown(f"- {trend}")
    with right:
        st.subheader("Clusters")
        for cluster in insights.get("clusters", []):
            st.markdown(
                f"- **{cluster.get('theme', 'Theme')}** ({cluster.get('document_count', 0)} docs): "
                f"{', '.join(cluster.get('examples', []))}"
            )
        st.subheader("Market Insight")
        st.write(insights.get("market_insight", ""))


if "research_result" not in st.session_state:
    st.session_state.research_result = None
if "report_result" not in st.session_state:
    st.session_state.report_result = None

query = st.text_input(
    "Research Query",
    value="Analyze emerging AI startups in healthcare.",
    help="Ask a market intelligence question to trigger search, vector indexing, and report generation.",
)

controls = st.columns([1, 1, 2])
run_clicked = controls[0].button("Run Research", use_container_width=True)
report_clicked = controls[1].button("Generate Report", use_container_width=True)
max_results = controls[2].slider("Max Sources", min_value=5, max_value=20, value=10, step=1)

if run_clicked:
    with st.spinner("Collecting sources, building embeddings, and extracting insights..."):
        try:
            st.session_state.research_result = post_json(
                "/research",
                {"query": query, "max_results": max_results},
            )
            st.session_state.report_result = None
        except httpx.HTTPError as exc:
            st.error(f"Research request failed: {exc}")

if report_clicked:
    collection_name = None
    if st.session_state.research_result:
        collection_name = st.session_state.research_result.get("collection_name")

    with st.spinner("Retrieving relevant documents and generating the report..."):
        try:
            st.session_state.report_result = post_json(
                "/report",
                {
                    "query": query,
                    "collection_name": collection_name,
                    "top_k": 8,
                },
            )
        except httpx.HTTPError as exc:
            st.error(f"Report request failed: {exc}")

if st.session_state.research_result:
    research = st.session_state.research_result
    st.success(
        f"Collected {research['source_count']} sources and stored {research['chunk_count']} chunks in "
        f"{research['mode']['vector_store']}."
    )
    meta_left, meta_right, meta_third = st.columns(3)
    meta_left.metric("Sources", research["source_count"])
    meta_right.metric("Chunks Indexed", research["chunk_count"])
    meta_third.metric("LLM Mode", research["mode"]["llm"])

    tab_sources, tab_insights = st.tabs(["Sources Collected", "Extracted Insights"])
    with tab_sources:
        render_sources(research.get("sources", []))
    with tab_insights:
        render_insights(research.get("insights", {}))

if st.session_state.report_result:
    report_result = st.session_state.report_result
    st.subheader("Final Report")
    st.markdown(report_result.get("report", ""))
    st.download_button(
        "Download Research Report",
        data=report_result.get("report", ""),
        file_name="research_report.md",
        mime="text/markdown",
    )
