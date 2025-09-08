from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import uuid
from laukik import upscale  # Assuming laukik.py is in the same directory

app = FastAPI()

# Define the path to the model.
# This assumes 'RealESRGAN_x4plus.pth' is in the same directory as main.py
MODEL_PATH = "RealESRGAN_x4plus.pth"

# Create a directory for temporary files if it doesn't exist
UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "outputs"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

@app.post("/upscale/")
async def upscale_image(file: UploadFile = File(...)):
    """
    Accepts an image file, upscales it using Real-ESRGAN, and returns the result.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    # Generate unique filenames for the temporary files
    file_id = str(uuid.uuid4())
    input_filename = f"{file_id}-{file.filename}"
    output_filename = f"upscaled-{input_filename}"
    
    input_path = os.path.join(UPLOADS_DIR, input_filename)
    output_path = os.path.join(OUTPUTS_DIR, output_filename)

    # Save the uploaded file
    try:
        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {e}")

    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail=f"Model file not found at {MODEL_PATH}")

    # Run the upscaling process
    try:
        # The upscale function from laukik.py is called here
        upscale(input_image_path=input_path, output_image_path=output_path, model_path=MODEL_PATH)
    except Exception as e:
        # Clean up the input file if upscaling fails
        os.remove(input_path)
        raise HTTPException(status_code=500, detail=f"Error during image upscaling: {e}")

    # Check if the output file was created
    if not os.path.exists(output_path):
        # Clean up the input file
        os.remove(input_path)
        raise HTTPException(status_code=500, detail="Upscaling process did not produce an output file.")

    # Return the upscaled image as a file response
    # The file will be automatically cleaned up by FileResponse
    return FileResponse(output_path, media_type="image/png", filename=output_filename, background_tasks=[lambda: os.remove(input_path), lambda: os.remove(output_path)])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Real-ESRGAN Upscaler API. Use the /docs endpoint to see the API documentation."}
