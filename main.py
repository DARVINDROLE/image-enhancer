from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
import cv2
import numpy as np
from PIL import Image
import os
import uuid

# Import Real-ESRGAN model
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

app = FastAPI(title="Image Upscaler API")

# Initialize model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upscaler = RealESRGANer(
    scale=4,
    model_path="RealESRGAN_x4plus.pth",  # Make sure to download this model
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True
)

@app.get("/")
def home():
    return {"message": "Welcome to the Image Upscaler API!"}

@app.post("/upscale/")
async def upscale_image(file: UploadFile = File(...)):
    # Save input image
    input_path = f"temp_{uuid.uuid4().hex}.png"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Load image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    # Upscale
    output, _ = upscaler.enhance(img, outscale=4)

    # Save result
    output_path = f"upscaled_{uuid.uuid4().hex}.png"
    cv2.imwrite(output_path, output)

    # Return file
    return FileResponse(output_path, media_type="image/png", filename="upscaled.png")
