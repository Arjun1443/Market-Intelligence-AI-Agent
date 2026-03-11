# Autonomous Market Intelligence AI Agent

An AI-powered research system that automatically collects web data, builds a knowledge base, and generates intelligent insights using multi-agent orchestration and Retrieval-Augmented Generation (RAG).

---

## Overview

The **Autonomous Market Intelligence AI Agent** is designed to automate online research and market intelligence. It gathers information from the web, processes it into a searchable knowledge base, and uses AI agents to generate insights from the collected data.

The system integrates **web scraping, vector search, and large language models** to provide structured research outputs for research, business intelligence, and market analysis.

---

## Features

- Autonomous web research using AI agents  
- Real-time web scraping and data collection  
- Semantic search using vector embeddings  
- Retrieval-Augmented Generation (RAG) pipeline  
- Multi-agent architecture for analysis and summarization  
- Interactive dashboard for queries and insights  
- Offline fallback mode when external APIs are unavailable  

---

## Architecture

The system follows a modular architecture combining web scraping, AI reasoning, and vector search.

Workflow:

1. User submits a research query.
2. The system collects relevant web data.
3. Content is cleaned and processed.
4. Data is converted into embeddings.
5. Embeddings are stored in a vector database.
6. RAG retrieves relevant information.
7. AI agents generate insights and summaries.
8. Results are displayed on the dashboard.

## Project Structure

```
ai-research-agent/
│
├── backend/
│   ├── main.py            # FastAPI backend API
│   ├── agents.py          # Multi-agent research logic
│   ├── scraper.py         # Web scraping using Apify
│   ├── embeddings.py      # Embedding generation
│   └── rag_pipeline.py    # RAG orchestration
│
├── frontend/
│   └── app.py             # Streamlit dashboard
│
├── config/
│   └── settings.py        # Environment configuration
│
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Technologies Used

**Programming Language**
- Python

**Frameworks**
- FastAPI
- Streamlit
- LangChain

**AI & Machine Learning**
- OpenAI API
- Retrieval-Augmented Generation (RAG)

**Data Collection**
- Apify

**Vector Database**
- Qdrant

**Tools**
- Uvicorn
- Python Virtual Environment

---

## Installation

### 1. Clone the repository
git clone https://github.com/nagaanweshreddy/ai-research-agent.git


The dashboard will open in your browser.

---

## Example Workflow

1. User enters a research question.
2. The system scrapes relevant online sources.
3. Content is processed and converted into embeddings.
4. Embeddings are stored in Qdrant.
5. RAG retrieves the most relevant knowledge.
6. AI agents analyze the information.
7. Insights are generated and displayed.

---

## Fallback Mode

The system can run even without external APIs.

- Without **APIFY_API_TOKEN** → demo research data is used.
- Without **OPENAI_API_KEY** → heuristic analysis and local embeddings are used.

This enables the application to run in **demo/offline mode**.

---

## Challenges

- Integrating multiple AI frameworks
- Handling noisy data from web scraping
- Designing a reliable RAG pipeline
- Managing vector storage and retrieval efficiently

---

## Future Improvements

- Real-time market monitoring dashboards
- Competitor intelligence automation
- Trend detection and predictive analytics
- Integration with financial and news APIs
- Cloud deployment and scalability
- Interactive analytics visualizations

