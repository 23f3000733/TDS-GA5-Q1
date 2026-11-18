# RAG Colosseum service

FastAPI retrieval service for the 50 passages in `knowledge_base.json`.

## Run locally

```sh
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

`POST /query` accepts `{"question":"...", "top_k":3}`. `GET /health` reports the loaded passage count. CORS allows all origins. Retrieval and answer extraction run locally without model downloads or external API keys, so normal requests are fast and concurrent workers can be added with Uvicorn.

## Deploy on Hugging Face Spaces

Create a Docker Space, upload this directory, and wait for the build to finish. The Space URL is the public base URL; the graded endpoint is that URL plus `/query`. The container listens on port 7860.

Example:

```sh
curl -X POST "$BASE_URL/query" -H 'Content-Type: application/json' -d '{"question":"What is the Raft quorum?","top_k":3}'
```
