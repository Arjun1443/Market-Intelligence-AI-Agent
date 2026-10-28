# Autonomous Market Intelligence AI Agent

A Python research application with a FastAPI backend, Streamlit dashboard, Apify web collection, and Qdrant retrieval. It can use OpenAI for embeddings and report generation, with local fallback behavior when API keys are absent.

## Quick start

The application lives in the `ai-research-agent/` subdirectory. Run these commands from a terminal:

```bash
git clone https://github.com/Arjun1443/Market-Intelligence-AI-Agent.git
cd Market-Intelligence-AI-Agent/ai-research-agent
python -m venv .venv
```

Activate the environment:

- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`

Install dependencies and start the backend from `ai-research-agent/`:

```bash
python -m pip install -r requirements.txt
python -m uvicorn main:app --app-dir backend --reload
```

In a second terminal, navigate to the same `ai-research-agent/` directory, activate the same environment, and start the dashboard:

```bash
python -m streamlit run frontend/app.py
```

Open the dashboard at http://localhost:8501. The API runs at http://localhost:8000, with interactive API documentation at http://localhost:8000/docs.

## Configuration and demo behavior

Create an optional `ai-research-agent/.env` file for live integrations:

```dotenv
OPENAI_API_KEY=
APIFY_API_TOKEN=
```

- Without `APIFY_API_TOKEN`, collection uses synthetic demo sources. These are placeholders, not current market evidence.
- Without `OPENAI_API_KEY`, analysis and embeddings use local fallback implementations.
- Qdrant defaults to local storage under `ai-research-agent/data/qdrant`. Set `QDRANT_URL` and optionally `QDRANT_API_KEY` for a server.
- The dashboard uses `http://localhost:8000` by default. Set the `API_BASE_URL` environment variable in the dashboard process to use another backend.
- Keep credentials out of commits.

See the [application documentation](ai-research-agent/README.md) and [configuration source](ai-research-agent/config/settings.py) for additional options.

## Try the application

1. Enter a research question, such as "Analyze emerging AI startups in healthcare."
2. Select **Run Research** to collect sources, index chunks, and inspect insights.
3. Select **Generate Report** to retrieve context and produce a report.
4. Download the Markdown report from the dashboard.

Check the backend independently:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/research -H "Content-Type: application/json" -d '{"query":"Analyze emerging AI startups in healthcare.","max_results":10}'
```

The health response includes `status`, `apify_configured`, and `openai_configured`. Request schemas and responses are available at `/docs`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `ai-research-agent/backend/main.py` | FastAPI endpoints: health, research, report |
| `ai-research-agent/backend/agents.py` | Research analysis and report agents |
| `ai-research-agent/backend/scraper.py` | Source collection |
| `ai-research-agent/backend/embeddings.py` | Embedding generation |
| `ai-research-agent/backend/rag_pipeline.py` | Research and retrieval pipeline |
| `ai-research-agent/frontend/app.py` | Streamlit dashboard |
| `ai-research-agent/config/settings.py` | Environment configuration |
| `ai-research-agent/requirements.txt` | Python dependencies |

## Limitations

Demo sources and heuristic outputs are intended to exercise the application, not support business decisions. Live integrations require valid service credentials and may incur usage charges. Validate source accuracy and generated claims before using a report.
