"""
Modal deployment for SeedVR2 Video Upscaler with FastAPI
Based on Flux Kontext pattern for better API design
"""

import modal
from pathlib import Path
from base64 import b64encode, b64decode
from io import BytesIO
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
    import cv2
    import numpy as np
    import subprocess
    import tempfile
    import sys


@app.cls(
    image=image,
    gpu="H100",
    volumes=volumes,
    timeout=1200,  # 20 minutes
    scaledown_window=300,
)
class SeedVR2Upscaler:
    @modal.enter()
    def enter(self):
        """Initialize on container startup"""
        print("Initializing SeedVR2 Upscaler...")
        
        # Set CUDA environment
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
        
        # Clone repo once at startup
        print("Cloning SeedVR2 repository...")
        subprocess.run(
            ["git", "clone", 
             "https://github.com/gkirilov7/ComfyUI-SeedVR2_VideoUpscaler.git", 
             "/root/seedvr2_repo"],
            check=True,
            capture_output=True,
        )
        
        print("SeedVR2 Upscaler ready!")
    
    @modal.method()
    def upscale(
        self,
        video_bytes: bytes,
        batch_size: int = 100,
        temporal_overlap: int = 12,
        stitch_mode: str = "crossfade",
        model: str = "seedvr2_ema_7b_fp16.safetensors"
    ) -> bytes:
        """
        Upscale video from bytes
        
        Args:
            video_bytes: Input video as bytes
            batch_size: Frames per batch
            temporal_overlap: Overlap frames between batches
            stitch_mode: 'later' or 'crossfade'
            model: Model to use
        
        Returns:
            Upscaled video as bytes
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write input video
            input_path = f"{tmpdir}/input.mp4"
            output_path = f"{tmpdir}/output.mp4"
            
            with open(input_path, 'wb') as f:
                f.write(video_bytes)
            
            input_size_mb = len(video_bytes) / (1024 * 1024)
            print(f"Processing {input_size_mb:.2f} MB video...")
            
            # Run inference
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
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/root/seedvr2_repo"
            )
            
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
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
    
    @modal.method()
    def upscale_from_url(self, video_url: str, batch_size: int = 100, temporal_overlap: int = 12, stitch_mode: str = "crossfade", model: str = "seedvr2_ema_7b_fp16.safetensors"):
        """Internal method that takes a URL and returns bytes"""
        import requests
        
        print(f"Downloading video from {video_url}...")
        response = requests.get(video_url, stream=True, timeout=300)
        response.raise_for_status()
        video_bytes = response.content
        
        return self.upscale.local(
            video_bytes,
            batch_size=batch_size,
            temporal_overlap=temporal_overlap,
            stitch_mode=stitch_mode,
            model=model
        )


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """FastAPI web interface"""
    from fastapi import FastAPI
    from pydantic import BaseModel
    
    web_app = FastAPI()
    
    class UpscaleRequest(BaseModel):
        video_url: str
        batch_size: int = 100
        temporal_overlap: int = 12
        stitch_mode: str = "crossfade"
        model: str = "seedvr2_ema_7b_fp16.safetensors"
    
    class UpscaleResponse(BaseModel):
        result: str
        input_size_mb: float
        output_size_mb: float
    
    @web_app.post("/generate", response_model=UpscaleResponse)
    def generate(request: UpscaleRequest):
        """
        Upscale video from URL
        
        Request body:
        {
            "video_url": "https://example.com/video.mp4",
            "batch_size": 100,
            "temporal_overlap": 12,
            "stitch_mode": "crossfade",
            "model": "seedvr2_ema_7b_fp16.safetensors"
        }
        
        Response:
        {
            "result": "base64-encoded output video",
            "input_size_mb": 10.5,
            "output_size_mb": 25.3
        }
        """
        import requests
        
        try:
            print(f"Downloading video from {request.video_url}...")
            response = requests.get(request.video_url, stream=True, timeout=300)
            response.raise_for_status()
            video_bytes = response.content
            
            input_size_mb = len(video_bytes) / (1024 * 1024)
            print(f"Processing {input_size_mb:.2f} MB video...")
            
            # Call upscaler
            upscaler = SeedVR2Upscaler()
            output_bytes = upscaler.upscale.remote(
                video_bytes,
                batch_size=request.batch_size,
                temporal_overlap=request.temporal_overlap,
                stitch_mode=request.stitch_mode,
                model=request.model
            )
            
            # Encode output
            output_b64 = b64encode(output_bytes).decode("utf-8")
            output_size_mb = len(output_bytes) / (1024 * 1024)
            
            return UpscaleResponse(
                result=output_b64,
                input_size_mb=input_size_mb,
                output_size_mb=output_size_mb
            )
        
        except Exception as e:
            print(f"Error: {str(e)}")
            raise
    
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
    upscaler = SeedVR2Upscaler()
    output_bytes = upscaler.upscale.remote(
        video_bytes,
        batch_size=batch_size,
        temporal_overlap=temporal_overlap,
        stitch_mode=stitch_mode,
        model=model
    )
    
    Path(output_path).write_bytes(output_bytes)
    print(f"Saved output to {output_path}")
