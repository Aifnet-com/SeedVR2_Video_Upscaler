"""
Async Modal deployment with persistent job tracking for SeedVR2 Video Upscaler
Uploads results directly to BunnyCDN - NO Modal volume commits!
Supports both URL and local file inputs with REAL-TIME LOG STREAMING via Modal Dict
Includes WATCHDOG to detect and kill stalled processes
"""

import modal
from typing import Dict, Optional
import json
from base64 import b64decode
import os

# Create Modal app
app = modal.App("seedvr2-upscaler")

# Create a shared dict for real-time progress updates (in-memory, fast)
progress_dict = modal.Dict.from_name("seedvr2-progress", create_if_missing=True)

# BunnyCDN Configuration
BUNNYCDN_API_KEY = os.environ.get("BUNNYCDN_API_KEY_VIDEO", "e084a0d7-30a9-49cb-8b355fd4f98a-da03-44cc")
BUNNYCDN_VIDEO_LIBRARY_ID = os.environ.get("BUNNYCDN_VIDEO_LIBRARY_ID", "131651")

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

# Create persistent volume for models only
model_volume = modal.Volume.from_name("seedvr2-models", create_if_missing=True)

# BunnyCDN Helper Functions
def create_bunny_video(title: str) -> Optional[str]:
    """Create a new video on BunnyCDN and return its GUID"""
    import requests
    import time

    url = f"https://video.bunnycdn.com/library/{BUNNYCDN_VIDEO_LIBRARY_ID}/videos"
    data = json.dumps({"title": title})
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "AccessKey": BUNNYCDN_API_KEY
    }

    for attempt in range(3):
        try:
            response = requests.post(url, data=data, headers=headers, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            return response_data.get('guid')
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Attempt {attempt + 1}/3 to create video failed: {e}")
            if attempt == 2:
                raise Exception(f"Failed to create video on BunnyCDN: {e}")
            time.sleep(2)

    return None


def upload_to_bunny(guid: str, file_path: str) -> dict:
    """Upload video file to BunnyCDN"""
    import requests
    import time

    url = f"https://video.bunnycdn.com/library/{BUNNYCDN_VIDEO_LIBRARY_ID}/videos/{guid}"
    headers = {
        'AccessKey': BUNNYCDN_API_KEY,
    }

    # Read file content
    with open(file_path, 'rb') as f:
        content = f.read()

    file_size_mb = len(content) / (1024 * 1024)
    print(f"📤 Uploading {file_size_mb:.2f} MB to BunnyCDN...")

    for attempt in range(3):
        try:
            response = requests.put(url, data=content, headers=headers, timeout=300)
            response.raise_for_status()

            print(f"✅ Upload successful!")

            return {
                'video_library_id': BUNNYCDN_VIDEO_LIBRARY_ID,
                'video_guid': guid,
                'file_size_mb': file_size_mb
            }
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Attempt {attempt + 1}/3 to upload failed: {e}")
            if attempt == 2:
                raise Exception(f"Failed to upload to BunnyCDN: {e}")
            time.sleep(2)

    return None


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    volumes={"/models": model_volume},
    scaledown_window=300,
    max_containers=10,
    secrets=[modal.Secret.from_dict({
        "BUNNYCDN_API_KEY_VIDEO": BUNNYCDN_API_KEY,
        "BUNNYCDN_VIDEO_LIBRARY_ID": BUNNYCDN_VIDEO_LIBRARY_ID
    })]
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
    volumes={"/models": model_volume},
    scaledown_window=300,
    max_containers=10,
    secrets=[modal.Secret.from_dict({
        "BUNNYCDN_API_KEY_VIDEO": BUNNYCDN_API_KEY,
        "BUNNYCDN_VIDEO_LIBRARY_ID": BUNNYCDN_VIDEO_LIBRARY_ID
    })]
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
    """Update job progress in Modal Dict"""
    if not job_id:
        return

    import time

    try:
        progress_dict[job_id] = {
            "text": progress_text,
            "timestamp": time.time()
        }
    except Exception as e:
        print(f"⚠️  Failed to update progress: {e}")


def _calculate_stall_timeout(resolution: str, batch_size: int = 100, total_frames: int = None) -> tuple:
    """
    Calculate stall timeout based on resolution and batch size
    Returns (first_batch_timeout, regular_batch_timeout)
    """

    # Base time per 100 frames in seconds (empirical data)
    time_per_100_frames = {
        '720p': 50,
        '1080p': 70,
        '2k': 120,
        '4k': 250,
    }

    base_time_per_100 = time_per_100_frames.get(resolution, 70)

    # Expected time for this batch size
    expected_batch_time = int(base_time_per_100 * (batch_size / 100))

    # Add overhead for model loading (first batch only)
    model_loading_overhead = 45

    # Calculate timeouts with safety margin (1.5x)
    first_batch_timeout = int((expected_batch_time + model_loading_overhead) * 1.5)
    regular_batch_timeout = int(expected_batch_time * 1.5)

    # Set minimum timeouts
    first_batch_timeout = max(first_batch_timeout, 180)  # At least 3 minutes
    regular_batch_timeout = max(regular_batch_timeout, 60)  # At least 1 minute

    print(f"📊 Stall timeout calculation:")
    print(f"   Resolution: {resolution}, Batch size: {batch_size}")
    print(f"   Base time per 100 frames: {base_time_per_100}s")
    print(f"   Expected time for {batch_size} frames: {expected_batch_time}s")
    print(f"   First batch timeout: {first_batch_timeout}s ({first_batch_timeout/60:.1f} min)")
    print(f"   Regular batch timeout: {regular_batch_timeout}s ({regular_batch_timeout/60:.1f} min)")

    return first_batch_timeout, regular_batch_timeout


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
    """Upscale a video using SeedVR2 from URL or base64 data with real-time progress updates and watchdog"""
    import subprocess
    import tempfile
    import os
    import requests
    import hashlib
    import time as time_module
    import shutil
    import cv2
    import math
    import threading
    import signal

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

        # Get video dimensions and duration
        _update_job_progress(job_id, "📐 Analyzing video dimensions...")
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = frame_count / fps if fps > 0 else 30
        cap.release()

        if width == 0 or height == 0:
            raise Exception(f"Could not read video dimensions: {width}x{height}")

        print(f"📐 Input dimensions: {width}x{height}")
        print(f"⏱️  Video duration: {video_duration:.1f} seconds ({frame_count} frames @ {fps:.2f} fps)")

        # Calculate stall timeout based on batch size, total frames, and resolution
        # This is critical: we only get updates BETWEEN batches!
        first_batch_timeout, regular_batch_timeout = _calculate_stall_timeout(resolution, batch_size, frame_count)

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
        new_width = width * ratio
        new_height = height * ratio

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
            "--resolution", str(int(resolution_px)),
            "--model_dir", "/models",
            "--output", temp_output_path,
            "--debug"
        ]

        print(f"🔧 Running upscaler with real-time logging and watchdog...")
        _update_job_progress(job_id, "🔧 Starting upscale process...")

        # Run with REAL-TIME OUTPUT STREAMING + WATCHDOG
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=repo_dir,
            preexec_fn=os.setsid
        )

        full_output = []
        last_progress_update = ""
        last_heartbeat = time_module.time()
        watchdog_killed = False
        is_first_batch = True
        current_stall_timeout = first_batch_timeout

        # Watchdog thread
        def watchdog():
            nonlocal watchdog_killed, last_heartbeat, is_first_batch, current_stall_timeout

            while process.poll() is None:
                elapsed = int(time_module.time() - last_heartbeat)
                timeout = current_stall_timeout

                batch_type = "FIRST_BATCH" if is_first_batch else "REGULAR"

                if elapsed % 10 == 0 and elapsed > 0:
                    print(f"🐕 Watchdog [{batch_type}]: {elapsed}s / {timeout}s")

                if elapsed > timeout:
                    print(f"⚠️  WATCHDOG: Process stalled for {elapsed}s (timeout: {timeout}s)")
                    print(f"⚠️  WATCHDOG: Killing process group...")
                    watchdog_killed = True
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except:
                        process.kill()
                    break

                time_module.sleep(1)

        watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        watchdog_thread.start()

        # Stream output
        for line in iter(process.stdout.readline, ''):
            if not line:
                break

            line = line.strip()
            if line:
                print(line)
                full_output.append(line)
                last_heartbeat = time_module.time()

                # Detect first batch completion and switch to faster timeout
                if "✅ First batch completed" in line or "First batch: OK" in line:
                    if is_first_batch:
                        print(f"✅ First batch completed - switching to faster watchdog timeout")
                        is_first_batch = False
                        current_stall_timeout = regular_batch_timeout

                # Extract and forward progress updates
                if any(keyword in line for keyword in ["Processing batch", "frames processed", "Upscaling", "🎬 Batch", "💾 Processing"]):
                    if line != last_progress_update:
                        _update_job_progress(job_id, line)
                        last_progress_update = line

        process.wait()

        if watchdog_killed:
            raise Exception(f"Process stalled for more than {current_stall_timeout}s and was killed by watchdog")

        if process.returncode != 0:
            error_msg = "\n".join(full_output[-20:])  # Last 20 lines
            print(f"❌ Error: {error_msg}")
            raise Exception(f"Upscaling failed: {error_msg}")

        if not os.path.exists(temp_output_path):
            raise Exception("Output file not created")

        output_size_mb = os.path.getsize(temp_output_path) / (1024 * 1024)
        print(f"✅ Output video size: {output_size_mb:.2f} MB")

        # Upload to BunnyCDN instead of Modal volume
        _update_job_progress(job_id, "📤 Uploading to CDN...")

        # Create video title
        video_title = f"upscaled_{url_hash}_{resolution}_{int(time_module.time())}"

        # Create video on BunnyCDN
        print(f"🎬 Creating video on BunnyCDN: {video_title}")
        guid = create_bunny_video(video_title)

        if not guid:
            raise Exception("Failed to create video on BunnyCDN")

        print(f"✅ Video created with GUID: {guid}")

        # Upload to BunnyCDN
        upload_result = upload_to_bunny(guid, temp_output_path)

        if not upload_result:
            raise Exception("Failed to upload to BunnyCDN")

        # Generate CDN URL
        cdn_url = f"https://vz-{BUNNYCDN_VIDEO_LIBRARY_ID}.b-cdn.net/{guid}/play_1080p.mp4"

        print(f"✅ CDN URL: {cdn_url}")
        _update_job_progress(job_id, "✅ Upload complete!")

    # Cleanup
    os.chdir("/root")
    shutil.rmtree(repo_dir, ignore_errors=True)

    return {
        "cdn_url": cdn_url,
        "video_guid": guid,
        "video_library_id": BUNNYCDN_VIDEO_LIBRARY_ID,
        "input_size_mb": input_size_mb,
        "output_size_mb": output_size_mb,
        "resolution": resolution
    }


@app.function(
    image=image,
    scaledown_window=300,
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI webhook for job submission and status checking"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uuid
    import time
    import asyncio

    web_app = FastAPI()

    JOBS_DIR = "/tmp/jobs"  # Use local temp storage instead of volume
    os.makedirs(JOBS_DIR, exist_ok=True)

    def save_job(job_id: str, job_data: dict):
        """Save job to local storage"""
        try:
            job_file = f"{JOBS_DIR}/{job_id}.json"
            with open(job_file, "w") as f:
                json.dump(job_data, f)
        except Exception as e:
            print(f"⚠️  Error saving job {job_id}: {e}")

    def load_job(job_id: str) -> Optional[dict]:
        """Load job from local storage"""
        job_file = f"{JOBS_DIR}/{job_id}.json"
        try:
            if os.path.exists(job_file):
                with open(job_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading job {job_id}: {e}")
        return None

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
        gpu_type: str

    class JobStatus(BaseModel):
        job_id: str
        status: str
        progress: Optional[str] = None
        cdn_url: Optional[str] = None
        video_guid: Optional[str] = None
        input_size_mb: Optional[float] = None
        output_size_mb: Optional[float] = None
        error: Optional[str] = None
        elapsed_seconds: Optional[float] = None

    def process_video(job_id: str, request: UpscaleRequest):
        """Process video upscaling job"""
        try:
            # Update job status
            job_data = load_job(job_id)
            if job_data:
                job_data["status"] = "processing"
                job_data["progress"] = "Starting upscaler..."
                save_job(job_id, job_data)

            print(f"[Job {job_id}] Starting upscale_video.remote()")

            # Select GPU function
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
                    job_id=job_id
                )
            else:
                print(f"[Job {job_id}] Using H200 for {request.resolution}")
                result = upscale_video_h200.remote(
                    video_url=request.video_url,
                    video_base64=request.video_base64,
                    batch_size=request.batch_size,
                    temporal_overlap=request.temporal_overlap,
                    stitch_mode=request.stitch_mode,
                    model=request.model,
                    resolution=request.resolution,
                    job_id=job_id
                )

            print(f"[Job {job_id}] Result: {result}")

            # Update job with result
            job_data = load_job(job_id)
            if job_data:
                job_data.update({
                    "status": "completed",
                    "cdn_url": result.get("cdn_url"),
                    "video_guid": result.get("video_guid"),
                    "input_size_mb": result.get("input_size_mb", 0),
                    "output_size_mb": result.get("output_size_mb", 0),
                    "progress": "✅ Completed and uploaded to CDN!"
                })
                save_job(job_id, job_data)

            # Clear progress dict
            try:
                if job_id in progress_dict:
                    del progress_dict[job_id]
            except:
                pass

            print(f"[Job {job_id}] ✅ Completed successfully")

        except Exception as e:
            print(f"[Job {job_id}] ❌ Failed: {str(e)}")
            job_data = load_job(job_id)
            if job_data:
                job_data.update({
                    "status": "failed",
                    "error": str(e),
                    "progress": f"❌ Failed: {str(e)}"
                })
                save_job(job_id, job_data)

            try:
                if job_id in progress_dict:
                    del progress_dict[job_id]
            except:
                pass

    @web_app.post("/upscale", response_model=JobResponse)
    async def upscale_endpoint(request: UpscaleRequest):
        """Submit a video upscaling job"""
        if not request.video_url and not request.video_base64:
            raise HTTPException(status_code=400, detail="Must provide either video_url or video_base64")

        job_id = str(uuid.uuid4())
        gpu_type = "H100" if request.resolution in ['720p', '1080p'] else "H200"

        job_data = {
            "job_id": job_id,
            "status": "pending",
            "created_at": time.time(),
            "gpu_type": gpu_type,
            "request": request.model_dump()
        }
        save_job(job_id, job_data)

        # Start processing in background
        asyncio.create_task(asyncio.to_thread(process_video, job_id, request))

        return {
            "job_id": job_id,
            "status": "pending",
            "message": f"Job submitted. Check status: GET /status/{job_id}",
            "gpu_type": gpu_type
        }

    @web_app.get("/status/{job_id}", response_model=JobStatus)
    async def get_job_status(job_id: str):
        """Check the status of a job"""

        # Check for real-time progress
        realtime_progress = None
        try:
            if job_id in progress_dict:
                progress_data = progress_dict[job_id]
                if isinstance(progress_data, dict):
                    realtime_progress = progress_data.get("text")
                else:
                    realtime_progress = progress_data
        except:
            pass

        job_data = load_job(job_id)
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")

        # Use real-time progress if available
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
            cdn_url=job_data.get("cdn_url"),
            video_guid=job_data.get("video_guid"),
            input_size_mb=job_data.get("input_size_mb"),
            output_size_mb=job_data.get("output_size_mb"),
            error=job_data.get("error"),
            elapsed_seconds=elapsed_seconds
        )

    @web_app.get("/")
    async def root():
        """API root"""
        return {
            "service": "SeedVR2 Video Upscaler",
            "version": "10.0 - BunnyCDN Direct Upload",
            "endpoints": {
                "submit_job": "POST /upscale",
                "check_status": "GET /status/{job_id}"
            },
            "features": {
                "bunnycdn_upload": "Results uploaded directly to BunnyCDN",
                "no_volume_commits": "No more Modal volume blocking issues",
                "watchdog": "Auto-kills stalled jobs",
                "real_time_logs": "Live progress updates",
                "cdn_delivery": "Global CDN for fast downloads"
            },
            "note": "Videos are uploaded to BunnyCDN - no download endpoint needed"
        }

    return web_app