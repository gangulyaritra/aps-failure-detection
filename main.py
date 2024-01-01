from fastapi import FastAPI, File, status, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, RedirectResponse
from uvicorn import run as app_run
from src.pipeline.training_pipeline import TrainPipeline
from src.pipeline.inference_pipeline import InferencePipeline


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
    try:
        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training Pipeline is already running.")

        train_pipeline.run_pipeline()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Training Pipeline Successful."},
        )

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/prediction_pipeline", response_description="Model Inference Pipeline.")
async def predict_route(file: UploadFile = File(...)):
    try:
        # Get the file size (in bytes).
        file.file.seek(0, 2)
        file_size = file.file.tell()

        # Move the cursor back to the beginning.
        await file.seek(0)

        if file_size > 30 * 1024 * 1024:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "File too large (> 30MB)."},
            )

        # Check the content type (MIME type).
        content_type = file.content_type
        if content_type not in ["text/csv"]:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "Invalid File Type. Please input a CSV file."},
            )

        inference_pipeline = InferencePipeline()
        if inference_pipeline.is_pipeline_running:
            return Response("Inference Pipeline is already running.")

        inference_pipeline.start_batch_prediction(file.file)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Inference Pipeline Successful."},
        )

    except Exception as e:
        return Response(f"Error Occurred! {e}")


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
