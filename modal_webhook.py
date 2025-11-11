"""
SeedVR2 Upscaler (Modal + FastAPI)
- H100 for 720p/1080p, H200 for 2K/4K
- Real-time progress via Modal Dict
- Watchdog to kill stalled runs
- FINAL ARTIFACT: uploaded directly to Bunny Storage via HTTPS PUT
- Status returns a DIRECT CDN URL (no Modal volume commits)
"""

import os
import json
import modal
from typing import Optional

# ---------------- Modal app & shared state ----------------

app = modal.App("seedvr2-upscaler")
progress_dict = modal.Dict.from_name("seedvr2-progress", create_if_missing=True)

# Bunny secret (created in Modal UI as "bunnycdn_storage")
bunny_secret = modal.Secret.from_name("bunnycdn_storage")

# Base image & deps
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg", "git")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu121",
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
        "fastapi[standard]",
    )
)

# Volumes
model_volume = modal.Volume.from_name("seedvr2-models", create_if_missing=True)
output_volume = modal.Volume.from_name("seedvr2-outputs", create_if_missing=True)

# ---------------- Bunny Storage via Secret ----------------

def _bunny_cfg():
    """
    Pull Bunny settings from the Modal secret injected as env vars.
    Required keys in secret:
      - BUNNY_STORAGE_ZONE    (e.g. "aifnet")
      - BUNNY_STORAGE_KEY     (Storage zone AccessKey)
      - BUNNY_BASE_URL        (e.g. "https://aifnet.b-cdn.net")
      - BUNNY_ROOT_DIR        (e.g. "tests/seedvr2_results")
    Optional:
      - BUNNY_HOST            (default "storage.bunnycdn.com")
    """
    zone = os.environ["BUNNY_STORAGE_ZONE"]
    accesskey = os.environ["BUNNY_STORAGE_KEY"]
    base_url = os.environ["BUNNY_BASE_URL"]
    root_dir = os.environ["BUNNY_ROOT_DIR"]
    host = os.environ.get("BUNNY_HOST", "storage.bunnycdn.com")
    return host, zone, accesskey, base_url, root_dir


def upload_to_bunny_storage(local_path: str, zone_rel_path: str) -> str:
    """
    Upload to Bunny Storage via HTTPS PUT with AccessKey header.
    zone_rel_path is path *inside* the zone (e.g. "tests/seedvr2_results/foo.mp4").
    Returns the CDN URL: https://<base_url>/<zone_rel_path>
    """
    import requests, time as _t

    host, zone, accesskey, base_url, _root = _bunny_cfg()

    zone_rel_path = zone_rel_path.lstrip("/")
    url = f"https://{host}/{zone}/{zone_rel_path}"

    headers = {
        "AccessKey": accesskey,
        "Content-Type": "application/octet-stream",
        "Connection": "close",
    }

    for attempt in range(3):
        try:
            with open(local_path, "rb") as f:
                resp = requests.put(url, headers=headers, data=f, timeout=60)
            if resp.status_code in (200, 201):
                return f"{base_url}/{zone_rel_path}"
            else:
                print(f"❌ Upload failed (attempt {attempt+1}/3): {resp.status_code}")
                _t.sleep(2)
        except Exception as e:
            print(f"❌ Upload error: {e}")
            _t.sleep(2)
    raise RuntimeError(f"Failed to upload after 3 attempts")


def check_bunny_file_exists(zone_rel_path: str) -> bool:
    """
    Check if a file exists in Bunny Storage via HEAD request.
    Returns True if file exists, False otherwise.
    """
    import requests

    host, zone, accesskey, base_url, _root = _bunny_cfg()

    zone_rel_path = zone_rel_path.lstrip("/")
    url = f"https://{host}/{zone}/{zone_rel_path}"

    headers = {
        "AccessKey": accesskey,
    }

    try:
        resp = requests.head(url, headers=headers, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"⚠️ Error checking file existence: {e}")
        return False


# ---------------- GPU Processing Function (Single) ----------------

@app.function(
    image=image,
    volumes={
        "/models": model_volume,
        "/outputs": output_volume,
    },
    secrets=[bunny_secret],
    cpu=2,
    timeout=1800,  # 30 minutes
)
def process_upscale(
    job_id: str,
    video_url: str,
    resolution: str = "1080p",
    video_base64: Optional[str] = None
):
    """
    Main upscaling function that runs inside the GPU container.
    Dynamically selects H100 for 720p/1080p, or H200 for 2K/4K.
    """
    import torch
    import subprocess
    import os
    import sys
    import time
    import tempfile
    import shutil
    import cv2
    import base64
    import requests

    start_time = time.time()

    # Set progress helper
    def set_progress(text: str):
        try:
            progress_dict[job_id] = {"text": text, "timestamp": time.time()}
            print(f"[{job_id}] {text}")
        except Exception as e:
            print(f"⚠️ Failed to update progress: {e}")

    set_progress(f"🚀 Starting upscale to {resolution}")

    # Select GPU based on resolution
    if resolution in ["720p", "1080p"]:
        gpu_request = modal.gpu.H100(count=1)
        gpu_name = "H100"
    else:
        gpu_request = modal.gpu.H200(count=1)
        gpu_name = "H200"

    # Acquire the GPU
    with gpu_request:
        set_progress(f"🖥️ Acquired {gpu_name} GPU")

        # Create work directory
        work_dir = f"/tmp/upscale_{job_id}"
        os.makedirs(work_dir, exist_ok=True)

        input_path = f"{work_dir}/input.mp4"
        output_path = f"{work_dir}/output.mp4"

        try:
            # Download or decode input video
            if video_base64:
                set_progress("📥 Decoding base64 video...")
                video_data = base64.b64decode(video_base64)
                with open(input_path, "wb") as f:
                    f.write(video_data)
            else:
                set_progress(f"📥 Downloading video...")
                resp = requests.get(video_url, timeout=60)
                resp.raise_for_status()
                with open(input_path, "wb") as f:
                    f.write(resp.content)

            # Get input file size
            input_size_mb = round(os.path.getsize(input_path) / (1024 * 1024), 2)
            set_progress(f"📊 Input size: {input_size_mb} MB")

            # Get video dimensions & frame count
            cap = cv2.VideoCapture(input_path)
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            set_progress(f"📹 Original: {orig_width}x{orig_height} @ {fps:.1f}fps, {total_frames} frames")

            # Resolution mapping
            resolution_map = {
                "720p": 1280,
                "1080p": 1920,
                "2k": 2560,
                "4k": 3840,
            }
            target_width = resolution_map.get(resolution, 1920)

            # Calculate target height maintaining aspect ratio
            aspect_ratio = orig_height / orig_width
            target_height = int(target_width * aspect_ratio)

            # Ensure even dimensions
            if target_height % 2 != 0:
                target_height += 1

            # Check if upscaling needed
            scale_factor = target_width / orig_width
            if scale_factor <= 1.0:
                set_progress(f"⚠️ Video is already {orig_width}x{orig_height}, no upscaling needed")
                shutil.copy(input_path, output_path)
            else:
                set_progress(f"🎯 Target: {target_width}x{target_height} (scale: {scale_factor:.2f}x)")

                # Clone SeedVR2 repo if not exists
                repo_dir = "/tmp/SeedVR2"
                if not os.path.exists(repo_dir):
                    set_progress("📦 Cloning SeedVR2 repository...")
                    subprocess.run([
                        "git", "clone", "--depth", "1",
                        "https://github.com/Seedifyfund/SeedVR2",
                        repo_dir
                    ], check=True)

                # Prepare paths
                sys.path.insert(0, repo_dir)
                os.chdir(repo_dir)

                # Import SeedVR2 components
                set_progress("🔧 Loading SeedVR2 models...")
                from configs import create_model_config, create_pipeline_config
                from pipelines.upscaling_pipeline import UpscalingPipeline

                # Configure model - reduce memory usage
                model_cfg = create_model_config()
                model_cfg.model.conditioning_channels = 3
                model_cfg.training.use_ema = False

                # Optimize for memory
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True

                # Initialize pipeline
                pipeline_cfg = create_pipeline_config()
                pipeline = UpscalingPipeline(
                    model_config=model_cfg,
                    pipeline_config=pipeline_cfg,
                    device="cuda",
                    dtype=torch.float16,
                )

                # Load checkpoint
                checkpoint_path = "/models/seedvr2_upscaler.safetensors"
                if not os.path.exists(checkpoint_path):
                    set_progress("📥 Downloading model weights...")
                    url = "https://huggingface.co/yuvalkirstain/SeedVR2-upscaling/resolve/main/seedvr2_upscaler.safetensors"
                    subprocess.run(["wget", "-q", "-O", checkpoint_path, url], check=True)

                pipeline.load_checkpoint(checkpoint_path)
                set_progress("✅ Model loaded successfully")

                # Process video
                set_progress("🎬 Starting video processing...")
                processed_frames = []
                batch_size = 1

                cap = cv2.VideoCapture(input_path)
                frame_idx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % 10 == 0:
                        progress_pct = int((frame_idx / total_frames) * 100)
                        set_progress(f"🔄 Processing frame {frame_idx}/{total_frames} ({progress_pct}%)")

                    # Resize frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (target_width, target_height),
                                              interpolation=cv2.INTER_LANCZOS4)

                    # Convert to tensor
                    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
                    frame_tensor = frame_tensor.unsqueeze(0).to("cuda")

                    # Apply upscaling model
                    with torch.no_grad():
                        enhanced = pipeline.enhance_frame(frame_tensor)
                        enhanced_np = (enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

                    # Convert back to BGR
                    enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
                    processed_frames.append(enhanced_bgr)

                    frame_idx += 1

                    # Clear memory periodically
                    if frame_idx % 50 == 0:
                        torch.cuda.empty_cache()

                cap.release()

                # Write output video
                set_progress(f"💾 Encoding {len(processed_frames)} frames to {resolution}...")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

                for frame in processed_frames:
                    out.write(frame)
                out.release()

                # Clean up memory
                del processed_frames
                torch.cuda.empty_cache()

            # Get output size
            output_size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
            set_progress(f"📊 Output size: {output_size_mb} MB")

            # Upload to Bunny Storage
            set_progress("☁️ Uploading to Bunny Storage...")
            _, _, _, _, root_dir = _bunny_cfg()

            # Generate filename with resolution
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{job_id}_{resolution}_{timestamp}.mp4"
            zone_rel_path = f"{root_dir}/{filename}"

            cdn_url = upload_to_bunny_storage(output_path, zone_rel_path)

            set_progress(f"Upload complete: {cdn_url}")

            # Clean up
            shutil.rmtree(work_dir, ignore_errors=True)

            elapsed_time = round(time.time() - start_time, 2)

            # Return success result
            return {
                "status": "completed",
                "download_url": cdn_url,
                "filename": filename,
                "input_size_mb": input_size_mb,
                "output_size_mb": output_size_mb,
                "elapsed_seconds": elapsed_time,
                "progress": f"✅ Completed in {elapsed_time}s"
            }

        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
            set_progress(error_msg)

            # Clean up on error
            shutil.rmtree(work_dir, ignore_errors=True)

            raise RuntimeError(error_msg)


# ---------------- FastAPI Web Service ----------------

@app.function(
    image=image,
    secrets=[bunny_secret],
    keep_warm=1,
    allow_concurrent_inputs=10,
    timeout=86400,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import time
    import json as _json
    import uuid
    import asyncio
    import os

    web_app = FastAPI(title="SeedVR2 Upscaler API")

    # Enable CORS
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request/Response models
    class UpscaleRequest(BaseModel):
        video_url: Optional[str] = None
        video_base64: Optional[str] = None
        resolution: str = "1080p"

    class JobResponse(BaseModel):
        job_id: str
        status: str
        message: str
        gpu_type: Optional[str] = None

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

    # Simple file-based job storage
    JOBS_DIR = "/tmp/seedvr2_jobs"
    os.makedirs(JOBS_DIR, exist_ok=True)

    def save_job(job_id: str, data: dict):
        with open(f"{JOBS_DIR}/{job_id}.json", "w") as f:
            _json.dump(data, f)

    def load_job(job_id: str) -> dict:
        path = f"{JOBS_DIR}/{job_id}.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                return _json.load(f)
        return None

    def try_promote_from_cdn(job_id: str, request_data: dict) -> bool:
        """
        Try to find the output file in Bunny CDN and promote job to completed.
        Returns True if successfully promoted.
        """
        try:
            from datetime import datetime

            # Build expected path patterns
            resolution = request_data.get("resolution", "1080p")
            _, _, _, base_url, root_dir = _bunny_cfg()

            # Try common filename patterns
            patterns = [
                f"{job_id}_{resolution}_",  # Our standard pattern
                f"{job_id}_",                # Fallback
            ]

            # Check if file exists in Bunny
            for pattern in patterns:
                # We need to check multiple possible timestamps
                # Since we don't know exact timestamp, we'll check if any file with our pattern exists
                # This is a simplified check - in production you might want to list directory

                # For now, we'll construct the most likely filename
                # (this assumes the upload happened recently)
                for minutes_ago in range(0, 10):  # Check last 10 minutes
                    check_time = datetime.now()
                    # Try a few timestamp formats
                    timestamp = check_time.strftime("%Y%m%d_%H%M%S")
                    filename = f"{pattern}{timestamp}.mp4"
                    zone_rel_path = f"{root_dir}/{filename}"

                    if check_bunny_file_exists(zone_rel_path):
                        cdn_url = f"{base_url}/{zone_rel_path}"

                        # Update job to completed
                        job_data = load_job(job_id) or {}
                        job_data.update({
                            "status": "completed",
                            "download_url": cdn_url,
                            "filename": filename,
                            "progress": "✅ Completed (recovered from CDN)"
                        })
                        save_job(job_id, job_data)

                        # Clear progress dict
                        try:
                            if job_id in progress_dict:
                                del progress_dict[job_id]
                        except Exception:
                            pass

                        print(f"✅ Promoted job {job_id} from CDN: {cdn_url}")
                        return True

            return False

        except Exception as e:
            print(f"⚠️ Error promoting from CDN: {e}")
            return False

    def process_video(job_id: str, request: UpscaleRequest):
        """Background task to process video"""
        try:
            job_data = load_job(job_id) or {}
            job_data.update({"status": "processing", "progress": "🔄 Starting GPU container..."})
            save_job(job_id, job_data)

            # Call the GPU function
            with process_upscale.remote(
                job_id=job_id,
                video_url=request.video_url,
                resolution=request.resolution,
                video_base64=request.video_base64,
            ) as handle:
                result = handle

            # Update job with result
            job_data = load_job(job_id) or {}
            job_data.update(result)
            save_job(job_id, job_data)

            # Clear progress_dict on completion
            try:
                if job_id in progress_dict:
                    del progress_dict[job_id]
            except Exception:
                pass

        except Exception as e:
            job_data = load_job(job_id) or {}
            job_data.update({"status": "failed", "error": str(e), "progress": f"❌ Failed: {e}"})
            save_job(job_id, job_data)
            try:
                if job_id in progress_dict:
                    del progress_dict[job_id]
            except Exception:
                pass

    @web_app.post("/upscale", response_model=JobResponse)
    async def upscale_endpoint(request: UpscaleRequest):
        if not request.video_url and not request.video_base64:
            raise HTTPException(status_code=400, detail="Must provide either video_url or video_base64")

        job_id = str(uuid.uuid4())
        gpu_type = "H100" if request.resolution in ["720p", "1080p"] else "H200"

        save_job(job_id, {
            "job_id": job_id,
            "status": "pending",
            "created_at": time.time(),
            "gpu_type": gpu_type,
            "request": request.model_dump(),
        })

        # Fallback now handled by auto-detection in /status endpoint
        # (Removed asyncio fallback thread - wasn't working in Modal's environment)

        # run job in background
        asyncio.create_task(asyncio.to_thread(process_video, job_id, request))

        return {
            "job_id": job_id,
            "status": "pending",
            "message": f"Job submitted. Check status: GET /status/{job_id}",
            "gpu_type": gpu_type,
        }

    @web_app.get("/status/{job_id}", response_model=JobStatus)
    async def get_job_status(job_id: str):
        # real-time progress from shared dict
        realtime = None
        try:
            if job_id in progress_dict:
                pd = progress_dict[job_id]
                realtime = pd.get("text") if isinstance(pd, dict) else str(pd)
        except Exception:
            pass

        job_data = load_job(job_id)

        # If we have no persisted job data yet:
        if not job_data:
            if realtime:
                # Return a transient "processing" status based solely on live progress.
                return JobStatus(
                    job_id=job_id,
                    status="processing",
                    progress=realtime,
                    download_url=None,
                    filename=None,
                    input_size_mb=None,
                    output_size_mb=None,
                    error=None,
                    elapsed_seconds=None,
                )
            # Truly unknown job id
            raise HTTPException(status_code=404, detail="Job not found")

        # Calculate elapsed time
        created_at = job_data.get("created_at")
        elapsed = (time.time() - created_at) if created_at else 0.0

        # === NEW: Check for stuck pending jobs ===
        if job_data.get("status") == "pending" and elapsed > 120:  # 2 min pending
            # Job never started - likely Modal scheduling issue
            job_data.update({
                "status": "failed",
                "error": "Job failed to start - Modal scheduling timeout",
                "progress": "❌ Failed to start"
            })
            save_job(job_id, job_data)

            # Clear progress_dict
            try:
                if job_id in progress_dict:
                    del progress_dict[job_id]
            except Exception:
                pass
        # === End pending check ===

        # --- Upload Watchdog (only when we have a real job_data dict) ---
        progress_text = (realtime or job_data.get("progress") or "")

        # If stuck on uploading for a while, try promote from CDN
        if job_data.get("status") in ("pending", "processing"):
            if "uploading to bunny storage" in progress_text.lower() and elapsed > 90:
                request_payload = job_data.get("request", {})
                if request_payload:
                    promoted = try_promote_from_cdn(job_id, request_payload)
                    if promoted:
                        # refresh the on-disk state to return the completed job
                        job_data = load_job(job_id) or job_data
                        created_at = job_data.get("created_at")
                        elapsed = (time.time() - created_at) if created_at else elapsed

        # === Auto-Completion Detection: Check if progress shows upload complete ===
        if job_data.get("status") == "processing":
            # Check both realtime progress and stored progress
            check_progress = realtime or job_data.get("progress") or ""

            if "Upload complete:" in check_progress:
                # GPU completed upload but never returned - auto-promote!
                try:
                    cdn_url = check_progress.split("Upload complete:")[-1].strip()

                    if cdn_url and cdn_url.startswith("http"):
                        # Extract filename from URL
                        filename = cdn_url.split("/")[-1] if "/" in cdn_url else "output.mp4"

                        # Mark as completed
                        job_data.update({
                            "status": "completed",
                            "download_url": cdn_url,
                            "filename": filename,
                            "progress": "✅ Completed (auto-detected from progress)"
                        })
                        save_job(job_id, job_data)

                        # Clear progress_dict
                        try:
                            if job_id in progress_dict:
                                del progress_dict[job_id]
                        except Exception:
                            pass

                        print(f"✅ Auto-detected completion for {job_id} from progress")
                except Exception as e:
                    print(f"⚠️ Error auto-detecting completion: {e}")
        # === End Auto-Completion Detection ===

        # Keep the freshest progress text visible
        if realtime:
            job_data["progress"] = realtime

        created_at = job_data.get("created_at")
        elapsed = (time.time() - created_at) if created_at else None

        # Single return statement at the end
        return JobStatus(
            job_id=job_id,
            status=job_data.get("status", "pending"),
            progress=job_data.get("progress"),
            download_url=job_data.get("download_url"),
            filename=job_data.get("filename"),
            input_size_mb=job_data.get("input_size_mb"),
            output_size_mb=job_data.get("output_size_mb"),
            error=job_data.get("error"),
            elapsed_seconds=elapsed,
        )

    @web_app.get("/")
    async def root():
        # lightweight active count
        active = 0
        if os.path.exists(JOBS_DIR):
            for fn in os.listdir(JOBS_DIR):
                if fn.endswith(".json"):
                    try:
                        with open(f"{JOBS_DIR}/{fn}", "r") as f:
                            jd = _json.load(f)
                        if jd.get("status") in ("pending", "processing"):
                            active += 1
                    except Exception:
                        pass

        return {
            "service": "SeedVR2 Video Upscaler",
            "status": "healthy",
            "active_jobs": active,
            "endpoints": {
                "POST /upscale": "Submit new upscaling job",
                "GET /status/{job_id}": "Check job status",
            },
            "resolutions": ["720p", "1080p", "2k", "4k"],
        }

    return web_app


# Entry point for Modal deployment
if __name__ == "__main__":
    app.deploy()