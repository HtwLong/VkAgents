from fastapi import APIRouter, HTTPException


router = APIRouter(tags=["Execution disabled"])
DETAIL = (
    "No execution of dataset download, preparation, training, evaluation, model loading, "
    "checkpoint generation, or inference is possible in this planning and results service."
)


async def disabled_operation():
    raise HTTPException(status_code=403, detail=DETAIL)


for method, path in (
    ("POST", "/download-data"),
    ("GET", "/download-data/status/{job_id}"),
    ("POST", "/prepare-data"),
    ("POST", "/evaluate"),
    ("POST", "/train/start"),
    ("GET", "/train/status/{job_id}"),
    ("GET", "/train/result/{job_id}"),
    ("POST", "/load-model"),
    ("POST", "/infer"),
    ("POST", "/unload-models"),
    ("POST", "/unload-model"),
):
    router.add_api_route(path, disabled_operation, methods=[method])
