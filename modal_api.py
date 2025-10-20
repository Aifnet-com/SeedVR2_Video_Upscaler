"""
Modal deployment for SeedVR2 Video Upscaler with FastAPI
"""

import modal
from pathlib import Path
from base64 import b64encode, b64decode
import os

# Build container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg", "git")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1", 
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "opencv-python-headless==4.10.0.84",
        "numpy>=1.26.4",
        "safetensors>=0.4.5",
        "einops",
        "omegaconf>=2.3.0",
        "diffusers>=0.34.0",
        "pytorch-extension",
        "rotary_embedding_torch",
        "peft>=0.15.0",
        "transformers>=4.46.3",
        "accelerate>=1.1.1",
        "huggingface-hub>=0.26.2",
        "requests>=2.32.3",
        "fastapi[standard]"
    )
)

# Persistent volumes
model_volume = modal.Volume.from_name("seedvr2-models", create_if_missing=True)
cache_dir = Path("/models")
volumes = {cache_dir: model_volume}

app = modal.App("seedvr2-upscaler")

with image.imports():
    import torch
    import subprocess
    import tempfile
    import sys
    import requests


@app.function(
    image=image,
    gpu="H100",
    volumes=volumes,
    timeout=1200,
)
def upscale_from_url(
    video_url: str,
    batch_size: int = 100,
    temporal_overlap: int = 12,
    stitch_mode: str = "crossfade",
    model: str = "seedvr2_ema_7b_fp16.safetensors"
) -> bytes:
    """
    Download video from URL, upscale it, and return bytes
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    
    # Clone repo once
    subprocess.run(
        ["git", "clone", 
         "https://github.com/gkirilov7/ComfyUI-SeedVR2_VideoUpscaler.git", 
         "/root/seedvr2_repo"],
        check=True,
        capture_output=True,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = f"{tmpdir}/input.mp4"
        output_path = f"{tmpdir}/output.mp4"
        
        # Download video
        print(f"Downloading video from {video_url}...")
        response = requests.get(video_url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(input_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        print(f"Input video size: {input_size_mb:.2f} MB")
        
        # Run upscaler
        cmd = [
            "python", "inference_cli.py",
            "--video_path", input_path,
            "--batch_size", str(batch_size),
            "--temporal_overlap", str(temporal_overlap),
            "--stitch_mode", stitch_mode,
            "--model", model,
            "--model_dir", str(cache_dir),
            "--output", output_path,
            "--debug"
        ]
        
        print(f"Running upscaler...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/root/seedvr2_repo"
        )
        
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            raise Exception(f"Upscaling failed: {result.stderr}")
        
        print(result.stdout)
        
        if not os.path.exists(output_path):
            raise Exception("Output file not created")
        
        # Read output
        with open(output_path, 'rb') as f:
            output_bytes = f.read()
        
        output_size_mb = len(output_bytes) / (1024 * 1024)
        print(f"Output size: {output_size_mb:.2f} MB")
        
        return output_bytes


@app.function(
    image=image.pip_install("fastapi[standard]"),
    timeout=1200,
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI web interface"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    web_app = FastAPI()
    
    class UpscaleRequest(BaseModel):
        video_url: str
        batch_size: int = 100
        temporal_overlap: int = 12
        stitch_mode: str = "crossfade"
        model: str = "seedvr2_ema_7b_fp16.safetensors"
    
    @web_app.post("/generate")
    def generate(request: UpscaleRequest):
        """
        Upscale video from URL
        
        Returns base64-encoded video in JSON response
        """
        try:
            print(f"Received request for {request.video_url}")
            
            # Call the upscale function
            output_bytes = upscale_from_url.remote(
                video_url=request.video_url,
                batch_size=request.batch_size,
                temporal_overlap=request.temporal_overlap,
                stitch_mode=request.stitch_mode,
                model=request.model
            )
            
            # Encode to base64
            output_b64 = b64encode(output_bytes).decode("utf-8")
            
            return {
                "result": output_b64,
                "output_size_mb": len(output_bytes) / (1024 * 1024)
            }
        
        except Exception as e:
            print(f"Error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.get("/")
    def root():
        return {"service": "SeedVR2 Upscaler", "endpoint": "/generate"}
    
    return web_app


@app.local_entrypoint()
def main(
    video_path: str,
    output_path: str = "output_upscaled.mp4",
    batch_size: int = 100,
    temporal_overlap: int = 12,
    stitch_mode: str = "crossfade",
    model: str = "seedvr2_ema_7b_fp16.safetensors"
):
    """
    Local test entrypoint
    
    Usage:
        modal run modal_app.py --video-path "input.mp4" --output-path "output.mp4"
    """
    print(f"Reading video from {video_path}")
    video_bytes = Path(video_path).read_bytes()
    
    print("Upscaling...")
    output_bytes = upscale_from_url.remote(
        video_url=None,  # Not used in local mode
        batch_size=batch_size,
        temporal_overlap=temporal_overlap,
        stitch_mode=stitch_mode,
        model=model
    )
    
    Path(output_path).write_bytes(output_bytes)
    print(f"Saved output to {output_path}")
