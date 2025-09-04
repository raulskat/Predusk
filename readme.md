Mini RAG Application
====================

Goal
- Build a minimal Retrieval-Augmented Generation (RAG) web app where users paste text, index it, then ask questions and get grounded answers with citations.

Project Structure
- `backend/`: FastAPI server exposing `/process-text` and `/query`.
- `frontend/index.html`: Single-file React UI.
- `.env.example`: Template for API keys and config.
- `requirements.txt`: Python deps (FastAPI, uvicorn, etc.).

Quickstart
- Prereqs: Python 3.10+ recommended.
- Create and activate a virtual env, then install deps:
  - `python -m venv .venv && .venv/Scripts/Activate.ps1` (PowerShell)
  - `pip install -r requirements.txt`
- Copy `.env.example` to `.env` and set values as needed.
  - Required: `PINECONE_API_KEY`, `PINECONE_INDEX_HOST` (or `PINECONE_INDEX` + `PINECONE_CLOUD` + `PINECONE_REGION`), `GOOGLE_API_KEY`, `COHERE_API_KEY`.
- Run the API:
  - `uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000`
- Open the frontend:
  - Open `frontend/index.html` directly, or serve the folder (e.g., `python -m http.server 5173 -d frontend`).
  - The UI defaults to `http://localhost:8000` for the API when on `localhost`.

API Endpoints
- `POST /process-text` — Chunk text and (placeholder) index. Body: `{ text, chunk_size, chunk_overlap }`
- `POST /query` — Ask a question. Body: `{ query, top_k }`

Current Status (Step 1)
- Backend and frontend scaffolded.
- Chunking implemented; embedding, vector DB, reranking, and LLM calls are placeholders for Step 2.
- Environment template added; README initialized.

Current Status (Step 2)
- Full RAG engine wired:
  - Embeddings via Gemini (`text-embedding-004`).
  - Storage/retrieval via Pinecone (`pinecone` SDK; set `PINECONE_INDEX_HOST` or let app create index using `PINECONE_INDEX` + cloud/region).
  - Reranking via Cohere (`rerank-english-v3.0`).
  - Answer generation via Gemini (`gemini-1.5-flash`).

Tech Choices
- Vector DB: Pinecone (use the `pinecone` package; `pinecone-client` is deprecated).
- Embeddings + LLM: Gemini.
- Reranking: Cohere.

Notes
- Google `google-generativeai` SDK is currently used; Google has introduced a new `google-genai` package. You can migrate later if desired.

Next Steps
- Implement Step 2 (RAG engine):
  - Embed chunks (Gemini), upsert to Pinecone with metadata.
  - Query Pinecone and rerank with Cohere.
  - Prompt Gemini and return answer + citations.
- Step 3: Harden API (logging, error handling) and wire actual engines.
- Step 4: Improve UI and render citations inline.
- Step 5: Deploy backend (Railway/Render) and frontend (Vercel/Netlify), finalize README.

Deployment
- Backend (Render):
  - Connect your GitHub repo on Render and pick this project.
  - Render will detect `render.yaml` and create a web service.
  - Set environment variables in Render dashboard:
    - `PINECONE_API_KEY`, `PINECONE_INDEX_HOST` (or `PINECONE_INDEX` + `PINECONE_CLOUD` + `PINECONE_REGION`), `GOOGLE_API_KEY`, `COHERE_API_KEY`, and `ALLOWED_ORIGINS` (set to your frontend URL).
  - Healthcheck at `/health` should return `{ "status": "ok" }`.
  - The API base URL will look like `https://mini-rag-backend.onrender.com`.

- Backend (Railway):
  - Use the included `Procfile`.
  - New project → Deploy from GitHub → set variables as above.
  - Set `PORT` if Railway requires manual setting; otherwise it injects it.

- Backend (Docker):
  - Build: `docker build -t mini-rag-backend .`
  - Run: `docker run -p 8000:8000 --env-file .env mini-rag-backend`

- Frontend (Vercel):
  - The repo contains `vercel.json` to serve `frontend/` as a static site.
  - Deploy via Vercel (import repo). No build step needed.
  - After deploy, open the site, set the API base URL in the UI (e.g., `https://<render-service>.onrender.com`).

- Frontend (Netlify):
  - The repo contains `netlify.toml` with `publish = "frontend"`.
  - Deploy via Netlify (import repo). No build step needed.

Production notes
- CORS: set `ALLOWED_ORIGINS` on the backend to your exact frontend origin (e.g., `https://mini-rag.vercel.app`). Comma-separate multiple origins.
- Secrets: never commit `.env`. Use provider dashboards to set secrets.
- Pinecone host: if you already have an index, set `PINECONE_INDEX_HOST` to avoid auto-creation.
- Cold start: first call to create/describe indexes may take a few seconds.
