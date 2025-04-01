from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from uvicorn import run as app_run

from aps.pipeline.inference_pipeline import InferencePipeline
from aps.pipeline.training_pipeline import TrainPipeline

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train_pipeline", response_description="Model Training Pipeline.")
async def train_route():
    train_pipeline = TrainPipeline()
    if train_pipeline.is_pipeline_running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training Pipeline is already running.",
        )

    try:
        train_pipeline.run_pipeline()
        return {"message": "Training Pipeline Successful."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error Occurred! {e}",
        ) from e


@app.post("/prediction_pipeline", response_description="Model Inference Pipeline.")
async def predict_route(file: UploadFile = File(...)):
    # Validate file size.
    file.file.seek(0, 2)
    file_size = file.file.tell()
    await file.seek(0)
    if file_size > 30 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File too large (> 30MB)"
        )

    # Validate file type.
    if file.content_type != "text/csv":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid File Type. Please input a CSV file.",
        )

    inference_pipeline = InferencePipeline()
    if inference_pipeline.is_pipeline_running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Inference Pipeline is already running.",
        )

    try:
        inference_pipeline.start_batch_prediction(file.file)
        return {"message": "Inference Pipeline Successful."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error Occurred! {e}",
        ) from e


def main():
    app_run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
