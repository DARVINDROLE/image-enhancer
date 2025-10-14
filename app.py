from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import uuid
from laukik import upscale  # Assuming laukik.py (your model code) is in the same directory

app = FastAPI(title="Real-ESRGAN Upscaler API")

MODEL_PATH = "RealESRGAN_x4plus.pth"
UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "outputs"

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Real-ESRGAN Upscaler API. Use /docs to test the endpoint."}

@app.post("/upscale/")
async def upscale_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Accepts an image file, upscales it using Real-ESRGAN, and returns the upscaled image.
    """
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only PNG and JPEG images are supported.")

    file_id = str(uuid.uuid4())
    input_filename = f"{file_id}_{file.filename}"
    output_filename = f"upscaled_{input_filename}"

    input_path = os.path.join(UPLOADS_DIR, input_filename)
    output_path = os.path.join(OUTPUTS_DIR, output_filename)

    # Save uploaded image
    try:
        with open(input_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {e}")

    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail=f"Model file not found at {MODEL_PATH}")

    try:
        upscale(input_image_path=input_path, output_image_path=output_path, model_path=MODEL_PATH)
    except Exception as e:
        os.remove(input_path)
        raise HTTPException(status_code=500, detail=f"Error during image upscaling: {e}")

    if not os.path.exists(output_path):
        os.remove(input_path)
        raise HTTPException(status_code=500, detail="Upscaling failed to produce an output image.")

    # Cleanup after sending
    background_tasks.add_task(os.remove, input_path)
    background_tasks.add_task(os.remove, output_path)

    return FileResponse(output_path, media_type="image/png", filename=output_filename)
