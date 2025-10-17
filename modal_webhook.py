"""
Complete Modal deployment with webhook for SeedVR2 Video Upscaler
"""

import modal

# Create Modal app
app = modal.App("seedvr2-upscaler")

# Define the container image with all dependencies
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

# Create persistent volumes
model_volume = modal.Volume.from_name("seedvr2-models", create_if_missing=True)
output_volume = modal.Volume.from_name("seedvr2-outputs", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",
    timeout=1000,
    volumes={"/models": model_volume},
    scaledown_window=300,
    max_containers=3,
)
@modal.concurrent(max_inputs=100)  # Add max_inputs parameter
def upscale_video(
    video_url: str,
    batch_size: int = 100,
    temporal_overlap: int = 12,
    stitch_mode: str = "crossfade",
    model: str = "seedvr2_ema_7b_fp16.safetensors"
):
    """Upscale a video using SeedVR2"""
    import subprocess
    import tempfile
    import os
    import requests
    
    print(f"🚀 Starting SeedVR2 upscaling")
    print(f"📋 Config: batch_size={batch_size}, overlap={temporal_overlap}, mode={stitch_mode}")
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    
    # Clone repo
    subprocess.run(["git", "clone", "https://github.com/gkirilov7/ComfyUI-SeedVR2_VideoUpscaler.git", "/root/repo"], check=True)
    os.chdir("/root/repo")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mp4")
        output_path = os.path.join(tmpdir, "output.mp4")
        
        print(f"📥 Downloading video from {video_url}")
        response = requests.get(video_url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(input_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        print(f"📦 Input video size: {input_size_mb:.2f} MB")
        
        cmd = [
            "python", "inference_cli.py",
            "--video_path", input_path,
            "--batch_size", str(batch_size),
            "--temporal_overlap", str(temporal_overlap),
            "--stitch_mode", stitch_mode,
            "--model", model,
            "--model_dir", "/models",
            "--output", output_path,
            "--debug"
        ]
        
        print(f"🔧 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/root/repo")
        
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            raise Exception(f"Upscaling failed: {result.stderr}")
        
        print(result.stdout)
        
        if not os.path.exists(output_path):
            raise Exception("Output file not created")
        
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Output video size: {output_size_mb:.2f} MB")
        
        with open(output_path, 'rb') as f:
            output_data = f.read()
        
        return {
            "video": output_data,
            "logs": result.stdout,
            "input_size_mb": input_size_mb,
            "output_size_mb": output_size_mb
        }


@app.function(
    image=image,
    volumes={"/outputs": output_volume},
    timeout=3600,
    scaledown_window=60,  # Changed from container_idle_timeout
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    import hashlib
    import time
    
    web_app = FastAPI()
    
    class UpscaleRequest(BaseModel):
        video_url: str
        batch_size: int = 100
        temporal_overlap: int = 12
        stitch_mode: str = "crossfade"
        model: str = "seedvr2_ema_7b_fp16.safetensors"
    
    @web_app.post("/upscale")
    async def upscale_endpoint(request: UpscaleRequest):
        """Upscale a video and return download URL"""
        try:
            # Call the upscale function
            result = upscale_video.remote(
                video_url=request.video_url,
                batch_size=request.batch_size,
                temporal_overlap=request.temporal_overlap,
                stitch_mode=request.stitch_mode,
                model=request.model
            )
            
            # Generate unique filename
            timestamp = int(time.time())
            url_hash = hashlib.md5(request.video_url.encode()).hexdigest()[:8]
            filename = f"upscaled_{timestamp}_{url_hash}.mp4"
            output_path = f"/outputs/{filename}"
            
            # Save to persistent storage
            with open(output_path, "wb") as f:
                f.write(result["video"])
            
            # Commit the volume
            output_volume.commit()
            
            # Return URL
            download_url = f"https://gkirilov7--seedvr2-upscaler-fastapi-app.modal.run/download/{filename}"
            
            return {
                "status": "success",
                "input_size_mb": result["input_size_mb"],
                "output_size_mb": result["output_size_mb"],
                "download_url": download_url,
                "filename": filename
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.get("/download/{filename}")
    async def download_video(filename: str):
        """Download an upscaled video"""
        file_path = f"/outputs/{filename}"
        
        import os
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            file_path,
            media_type="video/mp4",
            filename=filename
        )
    
    @web_app.get("/")
    async def root():
        return {
            "service": "SeedVR2 Video Upscaler",
            "version": "1.0",
            "endpoints": {
                "upscale": "POST /upscale",
                "download": "GET /download/{filename}"
            }
        }
    
    return web_app
