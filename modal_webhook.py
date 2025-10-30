"""
Async Modal deployment with persistent job tracking for SeedVR2 Video Upscaler
Supports both URL and local file inputs with REAL-TIME LOG STREAMING via Modal Dict
"""

import modal
from typing import Dict, Optional
import json
from base64 import b64decode

# Create Modal app
app = modal.App("seedvr2-upscaler")

# Create a shared dict for real-time progress updates (in-memory, fast)
progress_dict = modal.Dict.from_name("seedvr2-progress", create_if_missing=True)

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
    timeout=7200,
    volumes={
        "/models": model_volume,
        "/outputs": output_volume
    },
    scaledown_window=300,
    max_containers=10,
)
def upscale_video_h100(
    video_url: Optional[str] = None,
    video_base64: Optional[str] = None,
    batch_size: int = 100,
    temporal_overlap: int = 12,
    stitch_mode: str = "crossfade",
    model: str = "seedvr2_ema_7b_fp16.safetensors",
    resolution: str = "1080p",
    job_id: Optional[str] = None
):
    """Upscale a video using SeedVR2 from URL or base64 data (H100 for 720p/1080p)"""
    return _upscale_video_impl(
        video_url, video_base64, batch_size, temporal_overlap,
        stitch_mode, model, resolution, gpu_type="H100", job_id=job_id
    )


@app.function(
    image=image,
    gpu="H200",
    timeout=7200,
    volumes={
        "/models": model_volume,
        "/outputs": output_volume
    },
    scaledown_window=300,
    max_containers=10,
)
def upscale_video_h200(
    video_url: Optional[str] = None,
    video_base64: Optional[str] = None,
    batch_size: int = 100,
    temporal_overlap: int = 12,
    stitch_mode: str = "crossfade",
    model: str = "seedvr2_ema_7b_fp16.safetensors",
    resolution: str = "1080p",
    job_id: Optional[str] = None
):
    """Upscale a video using SeedVR2 from URL or base64 data (H200 for 2K/4K)"""
    return _upscale_video_impl(
        video_url, video_base64, batch_size, temporal_overlap,
        stitch_mode, model, resolution, gpu_type="H200", job_id=job_id
    )


def _update_job_progress(job_id: str, progress_text: str):
    """Update job progress in BOTH persistent storage AND in-memory dict for fast access"""
    if not job_id:
        return

    import os

    # Update in-memory dict (FAST - for real-time status checks)
    try:
        progress_dict[job_id] = progress_text
    except Exception as e:
        print(f"⚠️  Failed to update progress dict: {e}")

    # Also update persistent storage (slower but survives restarts)
    JOBS_DIR = "/outputs/jobs"
    job_file = f"{JOBS_DIR}/{job_id}.json"

    if os.path.exists(job_file):
        try:
            # Reload volume to see latest changes
            output_volume.reload()

            with open(job_file, "r") as f:
                job_data = json.load(f)

            job_data["progress"] = progress_text

            with open(job_file, "w") as f:
                json.dump(job_data, f)

            output_volume.commit()
        except Exception as e:
            print(f"⚠️  Failed to update progress file: {e}")


def _upscale_video_impl(
    video_url: Optional[str] = None,
    video_base64: Optional[str] = None,
    batch_size: int = 100,
    temporal_overlap: int = 12,
    stitch_mode: str = "crossfade",
    model: str = "seedvr2_ema_7b_fp16.safetensors",
    resolution: str = "1080p",
    gpu_type: str = "H100",
    job_id: Optional[str] = None
):
    """Upscale a video using SeedVR2 from URL or base64 data with real-time progress updates"""
    import subprocess
    import tempfile
    import os
    import requests
    import hashlib
    import time as time_module
    import shutil
    import cv2
    import math

    print(f"🚀 Starting SeedVR2 upscaling on {gpu_type}")
    print(f"📋 Config: batch_size={batch_size}, overlap={temporal_overlap}, mode={stitch_mode}, target={resolution}")

    _update_job_progress(job_id, "🚀 Initializing upscaler...")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

    # Clone repo to unique temp directory
    repo_dir = tempfile.mkdtemp(prefix="seedvr_")
    print(f"📂 Cloning repo to: {repo_dir}")
    _update_job_progress(job_id, "📂 Cloning repository...")

    try:
        subprocess.run(
            ["git", "clone", "https://github.com/Aifnet-com/SeedVR2_Video_Upscaler.git", repo_dir],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Git clone failed: {e.stderr}")
        raise

    os.chdir(repo_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mp4")
        temp_output_path = os.path.join(tmpdir, "output.mp4")

        # Download or decode video
        if video_url:
            print(f"📥 Downloading video from URL: {video_url}")
            _update_job_progress(job_id, "📥 Downloading video...")
            response = requests.get(video_url, stream=True, timeout=300)
            response.raise_for_status()

            with open(input_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Use URL for filename hash
            url_hash = hashlib.md5(video_url.encode()).hexdigest()[:8]
        elif video_base64:
            print(f"📥 Decoding video from base64")
            _update_job_progress(job_id, "📥 Decoding video...")
            video_bytes = b64decode(video_base64)
            with open(input_path, 'wb') as f:
                f.write(video_bytes)
            url_hash = hashlib.md5(video_base64.encode()).hexdigest()[:8]
        else:
            raise Exception("Must provide either video_url or video_base64")

        input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        print(f"📦 Input video size: {input_size_mb:.2f} MB")

        # Get video dimensions
        _update_job_progress(job_id, "📐 Analyzing video dimensions...")
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if width == 0 or height == 0:
            raise Exception(f"Could not read video dimensions: {width}x{height}")

        print(f"📐 Input dimensions: {width}x{height}")

        # Calculate target resolution based on input dimensions
        target_pixels_map = {
            '720p': 921600,   # 1280x720
            '1080p': 2073600, # 1920x1080
            '2k': 3686400,    # 2560x1440
            '4k': 8294400,    # 3840x2160
        }

        target_pixels = target_pixels_map.get(resolution, 2073600)

        # Initial scale based on target pixel count
        ratio = math.sqrt(target_pixels / (width * height))
        new_width = width * ratio      # Keep as float for precision
        new_height = height * ratio    # Keep as float for precision

        # Round BOTH dimensions to nearest multiple of 16
        new_width = round(new_width / 16) * 16
        new_height = round(new_height / 16) * 16

        # Use minimum side (short side) as resolution parameter
        resolution_px = min(new_width, new_height)

        print(f"📐 Calculated output: {new_width}x{new_height} (resolution={resolution_px}px)")
        print(f"   Target pixels: {target_pixels:,}")
        print(f"   Actual pixels: {new_width * new_height:,} ({((new_width * new_height / target_pixels - 1) * 100):+.1f}%)")

        cmd = [
            "python", "inference_cli.py",
            "--video_path", input_path,
            "--batch_size", str(batch_size),
            "--temporal_overlap", str(temporal_overlap),
            "--stitch_mode", stitch_mode,
            "--model", model,
            "--resolution", str(resolution_px),
            "--model_dir", "/models",
            "--output", temp_output_path,
            "--debug"
        ]

        print(f"🔧 Running upscaler with real-time logging...")
        _update_job_progress(job_id, "🔧 Starting upscale process...")

        # Run with REAL-TIME OUTPUT STREAMING
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            cwd=repo_dir
        )

        full_output = []
        last_progress_update = ""

        # Stream output line by line
        for line in iter(process.stdout.readline, ''):
            if not line:
                break

            line = line.rstrip()
            full_output.append(line)
            print(line)  # Still print to Modal logs

            # Extract interesting log lines for progress updates
            progress_update = None

            if "Window" in line and "-" in line:
                # Example: "🧩 Window 0-99 (len=100)"
                progress_update = line.strip()
            elif "Time batch:" in line:
                # Example: "🔄 Time batch: 107.80s"
                progress_update = line.strip()
            elif "Batch" in line and "/" in line:
                # Example: "Processing batch 1/3"
                progress_update = line.strip()
            elif "Loading model" in line or "Model loaded" in line:
                progress_update = "⚙️ " + line.strip()
            elif "Downloading" in line:
                progress_update = "⬇️ " + line.strip()
            elif "Processing" in line and "frames" in line:
                progress_update = "🎬 " + line.strip()

            # Update progress if we found something interesting
            if progress_update and progress_update != last_progress_update:
                _update_job_progress(job_id, progress_update)
                last_progress_update = progress_update

        # Wait for process to complete
        process.wait()

        if process.returncode != 0:
            error_msg = "\n".join(full_output[-20:])  # Last 20 lines
            print(f"❌ Error: {error_msg}")
            raise Exception(f"Upscaling failed: {error_msg}")

        if not os.path.exists(temp_output_path):
            raise Exception("Output file not created")

        output_size_mb = os.path.getsize(temp_output_path) / (1024 * 1024)
        print(f"✅ Output video size: {output_size_mb:.2f} MB")
        _update_job_progress(job_id, "✅ Finalizing output...")

        # Generate unique filename
        timestamp = int(time_module.time())
        filename = f"{url_hash}_{resolution}_{timestamp}.mp4"
        final_path = f"/outputs/{filename}"

        print(f"💾 Saving to: {final_path}")
        shutil.copy2(temp_output_path, final_path)

        # Force volume sync
        output_volume.commit()

        if not os.path.exists(final_path):
            raise Exception(f"Failed to save to volume: {final_path}")

        print(f"✅ Saved successfully: {filename}")

    # Cleanup
    os.chdir("/root")
    shutil.rmtree(repo_dir, ignore_errors=True)

    return {
        "filename": filename,
        "input_size_mb": input_size_mb,
        "output_size_mb": output_size_mb
    }


@app.function(
    image=image,
    volumes={"/outputs": output_volume},
    timeout=7200,
    scaledown_window=60,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    import time
    import uuid
    import os
    import asyncio

    web_app = FastAPI()

    # Persistent job storage
    JOBS_DIR = "/outputs/jobs"
    os.makedirs(JOBS_DIR, exist_ok=True)

    def save_job(job_id: str, job_data: dict):
        """Save job to persistent storage"""
        with open(f"{JOBS_DIR}/{job_id}.json", "w") as f:
            json.dump(job_data, f)
        output_volume.commit()

    def load_job(job_id: str) -> Optional[dict]:
        """Load job from persistent storage"""
        job_file = f"{JOBS_DIR}/{job_id}.json"
        if not os.path.exists(job_file):
            # Try reloading volume in case it was just created
            output_volume.reload()
        if not os.path.exists(job_file):
            return None
        with open(job_file, "r") as f:
            return json.load(f)

    class UpscaleRequest(BaseModel):
        video_url: Optional[str] = None
        video_base64: Optional[str] = None
        batch_size: int = 100
        temporal_overlap: int = 12
        stitch_mode: str = "crossfade"
        model: str = "seedvr2_ema_7b_fp16.safetensors"
        resolution: str = "1080p"

    class JobResponse(BaseModel):
        job_id: str
        status: str
        message: str
        gpu_type: str  # H100 or H200

    class JobStatus(BaseModel):
        job_id: str
        status: str
        progress: Optional[str] = None
        download_url: Optional[str] = None
        filename: Optional[str] = None
        input_size_mb: Optional[float] = None
        output_size_mb: Optional[float] = None
        error: Optional[str] = None
        elapsed_seconds: Optional[float] = None

    def process_video(job_id: str, request: UpscaleRequest):
        """Background task to process video"""
        try:
            job_data = load_job(job_id)
            if job_data:
                job_data["status"] = "processing"
                job_data["progress"] = "Starting upscaler..."
                save_job(job_id, job_data)

            print(f"[Job {job_id}] Starting upscale_video.remote()")

            # Select GPU based on resolution and pass job_id for progress updates
            if request.resolution in ['720p', '1080p']:
                print(f"[Job {job_id}] Using H100 for {request.resolution}")
                result = upscale_video_h100.remote(
                    video_url=request.video_url,
                    video_base64=request.video_base64,
                    batch_size=request.batch_size,
                    temporal_overlap=request.temporal_overlap,
                    stitch_mode=request.stitch_mode,
                    model=request.model,
                    resolution=request.resolution,
                    job_id=job_id  # Pass job_id for real-time updates!
                )
            else:  # 2k or 4k
                print(f"[Job {job_id}] Using H200 for {request.resolution}")
                result = upscale_video_h200.remote(
                    video_url=request.video_url,
                    video_base64=request.video_base64,
                    batch_size=request.batch_size,
                    temporal_overlap=request.temporal_overlap,
                    stitch_mode=request.stitch_mode,
                    model=request.model,
                    resolution=request.resolution,
                    job_id=job_id  # Pass job_id for real-time updates!
                )

            filename = result["filename"]
            print(f"[Job {job_id}] Upscale completed: {filename}")

            # Verify file exists
            final_path = f"/outputs/{filename}"
            output_volume.reload()

            if not os.path.exists(final_path):
                raise Exception(f"Output file not found: {final_path}")

            download_url = f"https://aifnet--seedvr2-upscaler-fastapi-app.modal.run/download/{filename}"

            job_data = load_job(job_id)
            if job_data:
                job_data.update({
                    "status": "completed",
                    "download_url": download_url,
                    "filename": filename,
                    "input_size_mb": result["input_size_mb"],
                    "output_size_mb": result["output_size_mb"],
                    "progress": "✅ Completed successfully!"
                })
                save_job(job_id, job_data)

            # Clear from in-memory dict after completion
            try:
                if job_id in progress_dict:
                    del progress_dict[job_id]
            except:
                pass

            print(f"[Job {job_id}] Status saved")

        except Exception as e:
            print(f"[Job {job_id}] Error: {str(e)}")
            job_data = load_job(job_id)
            if job_data:
                job_data.update({
                    "status": "failed",
                    "error": str(e),
                    "progress": f"❌ Failed: {str(e)}"
                })
                save_job(job_id, job_data)

            # Clear from in-memory dict after failure
            try:
                if job_id in progress_dict:
                    del progress_dict[job_id]
            except:
                pass

    @web_app.post("/upscale", response_model=JobResponse)
    async def upscale_endpoint(request: UpscaleRequest):
        """Submit a video upscaling job (returns immediately with job ID)"""
        if not request.video_url and not request.video_base64:
            raise HTTPException(status_code=400, detail="Must provide either video_url or video_base64")

        job_id = str(uuid.uuid4())

        # Determine GPU based on resolution
        gpu_type = "H100" if request.resolution in ['720p', '1080p'] else "H200"

        job_data = {
            "job_id": job_id,
            "status": "pending",
            "created_at": time.time(),
            "gpu_type": gpu_type,
            "request": request.model_dump()
        }
        save_job(job_id, job_data)

        # Spawn in background thread
        asyncio.create_task(asyncio.to_thread(process_video, job_id, request))

        return {
            "job_id": job_id,
            "status": "pending",
            "message": f"Job submitted. Check status: GET /status/{job_id}",
            "gpu_type": gpu_type
        }

    @web_app.get("/status/{job_id}", response_model=JobStatus)
    async def get_job_status(job_id: str):
        """Check the status of a job - checks in-memory dict FIRST for real-time progress"""

        # First check the fast in-memory dict for real-time progress
        realtime_progress = None
        try:
            if job_id in progress_dict:
                realtime_progress = progress_dict[job_id]
        except:
            pass

        # Then load from persistent storage
        job_data = load_job(job_id)
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")

        # Use real-time progress if available, otherwise fall back to stored progress
        if realtime_progress:
            job_data["progress"] = realtime_progress

        created_at = job_data.get("created_at")
        elapsed_seconds = None
        if created_at:
            elapsed_seconds = time.time() - created_at

        return JobStatus(
            job_id=job_id,
            status=job_data["status"],
            progress=job_data.get("progress"),
            download_url=job_data.get("download_url"),
            filename=job_data.get("filename"),
            input_size_mb=job_data.get("input_size_mb"),
            output_size_mb=job_data.get("output_size_mb"),
            error=job_data.get("error"),
            elapsed_seconds=elapsed_seconds
        )

    @web_app.get("/download/{filename}")
    async def download_video(filename: str):
        """Download an upscaled video"""
        file_path = f"/outputs/{filename}"

        if not os.path.exists(file_path):
            output_volume.reload()

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        return FileResponse(
            file_path,
            media_type="video/mp4",
            filename=filename
        )

    @web_app.get("/")
    async def root():
        active_count = 0
        if os.path.exists(JOBS_DIR):
            for job_filename in os.listdir(JOBS_DIR):
                if job_filename.endswith('.json'):
                    job_data = load_job(job_filename[:-5])
                    if job_data and job_data["status"] in ["pending", "processing"]:
                        active_count += 1

        return {
            "service": "SeedVR2 Video Upscaler",
            "version": "3.3 (H100 for 720p/1080p, H200 for 2K/4K) - REAL-TIME LOGS via Modal Dict",
            "endpoints": {
                "submit_job": "POST /upscale",
                "check_status": "GET /status/{job_id}",
                "download": "GET /download/{filename}"
            },
            "active_jobs": active_count
        }

    return web_app