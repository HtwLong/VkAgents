# VKG planning and results backend

This is the deployable, CPU-only companion to `backend/`. It can read exported
run evidence, execute lightweight LLM planning, and create post-training
assessments. New planning can use the bundled metadata-only GraphRAG ontology.
Dataset download, preparation, training, evaluation execution,
checkpoint generation, model loading, and inference are intentionally blocked.

GraphRAG reads local CSV nodes/edges and performs deterministic candidate,
domain, memory, and recipe-bound filtering. It does not import model classes,
load weights, inspect CUDA, or read images. The package has no PyTorch
dependency. A deployment should contain only this
directory, not the original `backend/` directory.

## Local development

```bash
cd viewer_backend
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
OPENAI_API_KEY=... .venv/bin/uvicorn viewer_backend.api:app --reload
```

The default run directory is `viewer_backend/runs`. Override it with
`VIEWER_RUNS_DIR`. Generated planning state is written there too, so configure
external/persistent storage before relying on generated runs on an ephemeral
host.

Export compact historical evidence from the full backend with:

```bash
python scripts/export_runs.py ../backend/runs runs
```

## Hosted mode

Recommended Render command:

```bash
uvicorn viewer_backend.api:app --host 0.0.0.0 --port $PORT
```

Relevant environment variables are documented in `.env.example`.

## Existing frontend

The existing `../frontend` is the viewer UI too; a separate frontend is not
needed. For a Vercel deployment set:

```text
BACKEND_INTERNAL_URL=https://your-viewer-service.onrender.com
NEXT_PUBLIC_DEPLOYMENT_MODE=viewer
```

Leave `NEXT_PUBLIC_API_BASE_URL` unset so the browser uses the same-origin
`/api/backend` rewrite. The UI also reads `/api/v1/capabilities` and removes
execution, inference, and revision-execution controls when connected
to this service.
