# Autonomous Market Intelligence AI Agent

Full-stack research system that collects market signals from the web, stores embeddings in Qdrant, performs retrieval-augmented analysis, and serves the results through a Streamlit dashboard.

## Architecture

- Frontend: Streamlit dashboard
- Backend: FastAPI
- AI orchestration: LangChain
- Vector database: Qdrant
- Web data collection: Apify API
- LLM: OpenAI

## Project Structure

```text
ai-research-agent/
├── backend/
│   ├── agents.py
│   ├── embeddings.py
│   ├── main.py
│   ├── rag_pipeline.py
│   └── scraper.py
├── config/
│   └── settings.py
├── frontend/
│   └── app.py
├── requirements.txt
└── README.md
```

## Features

- Research query input
- Automated web data collection via Apify
- Embedding generation and Qdrant indexing
- Semantic retrieval for RAG
- Trend, entity, and clustering analysis
- Structured market research report generation
- Downloadable Markdown report in Streamlit

## Environment Variables

Create a `.env` file in the project root if you want live integrations:

```env
OPENAI_API_KEY=your_openai_key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

APIFY_API_TOKEN=your_apify_token
APIFY_ACTOR_ID=apify/google-search-scraper
APIFY_COUNTRY_CODE=us
APIFY_LANGUAGE_CODE=en

QDRANT_URL=
QDRANT_API_KEY=
QDRANT_LOCAL_PATH=./data/qdrant
```

## Install

```bash
pip install -r requirements.txt
```

## Run The Backend

From `ai-research-agent/backend`:

```bash
uvicorn main:app --reload
```

## Run The Frontend

From `ai-research-agent/frontend`:

```bash
streamlit run app.py
```

## API Endpoints

- `GET /health`
- `POST /research`
- `POST /report`

## Example Research Request

```json
{
  "query": "Analyze emerging AI startups in healthcare.",
  "max_results": 10
}
```

## Agent Design

### 1. Search Agent

- Calls the Apify actor
- Collects structured web search data
- Normalizes titles, snippets, URLs, and source types

### 2. Analysis Agent

- Extracts likely entities
- Identifies recurring trends
- Clusters similar sources
- Produces market insight and future opportunity signals

### 3. Report Agent

- Retrieves top documents from Qdrant
- Applies RAG context to report generation
- Produces the final Markdown report sections:
  - Key Companies
  - Emerging Trends
  - Market Insights
  - Future Opportunities

## Notes

- If `APIFY_API_TOKEN` is missing, the app runs in transparent demo mode with synthetic placeholder sources so the pipeline can still be tested locally.
- If `OPENAI_API_KEY` is missing, the app falls back to heuristic analysis and a lightweight local embedding implementation so the system still runs end to end.
- For production use, point `QDRANT_URL` at a managed or self-hosted Qdrant instance instead of local disk mode.
