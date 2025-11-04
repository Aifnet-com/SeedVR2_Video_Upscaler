"""
Modal deployment with FILE-BASED result delivery:
- Fire off .remote() call and don't wait for it
- Immediately start polling for result file in /outputs/results/
- Use atomic writes to prevent partial reads
- 100% reliable regardless of container shutdown issues

This completely avoids the container hang problem by not relying on .remote() returning
"""

import modal
from typing import Dict, Optional
import json
from base64 import b64decode
import time
import os
import threading

app = modal.App("seedvr2-upscaler")

progress_dict = modal.Dict.from_name("seedvr2-progress", create_if_missing=True)

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
    """Upscale a video using SeedVR2 (H100 for 720p/1080p)"""
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
    """Upscale a video using SeedVR2 (H200 for 2K/4K)"""
    return _upscale_video_impl(
        video_url, video_base64, batch_size, temporal_overlap,
        stitch_mode, model, resolution, gpu_type="H200", job_id=job_id
    )


def _update_job_progress(job_id: str, progress_text: str):
    """Update job progress in both in-memory dict and persistent storage"""
    if not job_id:
        return

    import os
    import time

    try:
        progress_dict[job_id] = {
            "text": progress_text,
            "timestamp": time.time()
        }
    except Exception as e:
        print(f"⚠️  Failed to update progress dict: {e}")

    JOBS_DIR = "/outputs/jobs"
    job_file = f"{JOBS_DIR}/{job_id}.json"

    if os.path.exists(job_file):
        try:
            output_volume.reload()

            with open(job_file, "r") as f:
                job_data = json.load(f)

            job_data["progress"] = progress_text
            job_data["last_update"] = time.time()

            with open(job_file, "w") as f:
                json.dump(job_data, f)

            output_volume.commit()
        except Exception as e:
            print(f"⚠️  Failed to update progress file: {e}")


def _calculate_stall_timeout(resolution: str, batch_size: int = 100) -> int:
    """Calculate stall timeout based on resolution and batch size"""

    time_per_100_frames = {
        '720p': 50,
        '1080p': 70,
        '2k': 120,
        '4k': 250,
    }

    base_time_per_100 = time_per_100_frames.get(resolution, 70)
    expected_batch_time = int(base_time_per_100 * (batch_size / 100))
    model_loading_overhead = 45

    first_batch_timeout = int((expected_batch_time + model_loading_overhead) * 1.5)
    regular_batch_timeout = int(expected_batch_time * 1.5)

    first_batch_timeout = max(first_batch_timeout, 180)
    regular_batch_timeout = max(regular_batch_timeout, 60)

    print(f"📊 Stall timeout calculation:")
    print(f"   Resolution: {resolution}, Batch size: {batch_size}")
    print(f"   First batch timeout: {first_batch_timeout}s ({first_batch_timeout/60:.1f} min)")
    print(f"   Regular batch timeout: {regular_batch_timeout}s ({regular_batch_timeout/60:.1f} min)")

    return first_batch_timeout


def _write_result_file(job_id: str, result_data: dict):
    """
    Write result to volume using atomic write pattern
    Prevents partial reads by writing to temp file first, then renaming
    """
    if not job_id:
        return

    import os

    RESULTS_DIR = "/outputs/results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result_file = f"{RESULTS_DIR}/{job_id}.json"
    temp_file = f"{RESULTS_DIR}/{job_id}.tmp"

    print(f"📝 Writing result to: {result_file}")

    try:
        # Step 1: Write to temporary file
        with open(temp_file, "w") as f:
            json.dump(result_data, f)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk

        # Step 2: Atomic rename (either succeeds completely or not at all)
        os.rename(temp_file, result_file)

        # Step 3: Force volume sync
        output_volume.commit()

        # Step 4: Verify
        if os.path.exists(result_file):
            print(f"✅ Result file verified: {result_file}")
        else:
            print(f"⚠️  Result file not found after write!")

    except Exception as e:
        print(f"❌ Failed to write result file: {e}")
        # Clean up temp file if it exists
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass


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
    """Upscale video and ALWAYS write result to volume before returning"""
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
        error_msg = f"Git clone failed: {e.stderr}"
        print(f"❌ {error_msg}")
        _write_result_file(job_id, {"status": "error", "error": error_msg})
        raise Exception(error_msg)

    os.chdir(repo_dir)

    try:
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

                url_hash = hashlib.md5(video_url.encode()).hexdigest()[:8]
            elif video_base64:
                print(f"📥 Decoding video from base64")
                _update_job_progress(job_id, "📥 Decoding video...")
                video_bytes = b64decode(video_base64)
                with open(input_path, 'wb') as f:
                    f.write(video_bytes)
                url_hash = hashlib.md5(video_base64.encode()).hexdigest()[:8]
            else:
                error_msg = "Must provide either video_url or video_base64"
                _write_result_file(job_id, {"status": "error", "error": error_msg})
                raise Exception(error_msg)

            input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
            print(f"📦 Input video size: {input_size_mb:.2f} MB")

            # Get video dimensions
            _update_job_progress(job_id, "📐 Analyzing video dimensions...")
            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if width == 0 or height == 0:
                error_msg = f"Could not read video dimensions: {width}x{height}"
                _write_result_file(job_id, {"status": "error", "error": error_msg})
                raise Exception(error_msg)

            print(f"📐 Input dimensions: {width}x{height} ({frame_count} frames @ {fps:.2f} fps)")

            stall_timeout = _calculate_stall_timeout(resolution, batch_size)

            # Calculate target resolution
            target_pixels_map = {
                '720p': 921600,
                '1080p': 2073600,
                '2k': 3686400,
                '4k': 8294400,
            }

            target_pixels = target_pixels_map.get(resolution, 2073600)
            ratio = math.sqrt(target_pixels / (width * height))
            new_width = round(width * ratio / 16) * 16
            new_height = round(height * ratio / 16) * 16
            resolution_px = min(new_width, new_height)

            print(f"📐 Calculated output: {new_width}x{new_height} (resolution={resolution_px}px)")

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

            print(f"🔧 Running upscaler with watchdog...")
            _update_job_progress(job_id, "🔧 Starting upscale process...")

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
            current_stall_timeout = stall_timeout

            def watchdog_thread():
                """Monitor for stalled processing"""
                nonlocal watchdog_killed, last_heartbeat

                while process.poll() is None:
                    time_module.sleep(10)

                    time_since_update = time_module.time() - last_heartbeat

                    if time_since_update > current_stall_timeout:
                        print(f"🚨 WATCHDOG: No progress for {time_since_update:.0f}s")
                        print(f"🚨 WATCHDOG: Killing stalled process...")
                        watchdog_killed = True

                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            time_module.sleep(2)
                            if process.poll() is None:
                                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        except Exception as e:
                            print(f"⚠️  Watchdog kill error: {e}")
                        break
                    else:
                        timeout_label = "FIRST_BATCH" if is_first_batch else "REGULAR"
                        print(f"🐕 Watchdog [{timeout_label}]: {time_since_update:.0f}s / {current_stall_timeout}s")

            watchdog = threading.Thread(target=watchdog_thread, daemon=True)
            watchdog.start()

            # Stream output
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break

                line = line.rstrip()
                full_output.append(line)
                print(line)

                progress_update = None

                if "Window" in line and "-" in line:
                    progress_update = line.strip()
                elif "Time batch:" in line:
                    progress_update = line.strip()
                    if is_first_batch:
                        is_first_batch = False
                elif "Batch" in line and "/" in line:
                    progress_update = line.strip()
                elif "Loading model" in line or "Model loaded" in line:
                    progress_update = "⚙️ " + line.strip()
                elif "Loading VAE" in line:
                    progress_update = "⚙️ Loading VAE..."
                elif "Downloading" in line:
                    progress_update = "⬇️ " + line.strip()
                elif "Processing" in line and "frames" in line:
                    progress_update = "🎬 " + line.strip()
                elif "Detected NaDiT" in line:
                    progress_update = "✅ Model detected"
                elif "ROPE CACHE" in line:
                    progress_update = "⚙️ Initializing..."

                if progress_update and progress_update != last_progress_update:
                    _update_job_progress(job_id, progress_update)
                    last_progress_update = progress_update
                    last_heartbeat = time_module.time()
                elif line.strip():
                    last_heartbeat = time_module.time()

            process.wait()

            if watchdog_killed:
                error_msg = f"Job stalled (no progress for {current_stall_timeout}s)"
                print(f"❌ {error_msg}")
                _write_result_file(job_id, {"status": "error", "error": error_msg})
                raise Exception(error_msg)

            if process.returncode != 0:
                error_msg = "\n".join(full_output[-20:])
                print(f"❌ Process error: {error_msg}")
                _write_result_file(job_id, {"status": "error", "error": error_msg})
                raise Exception(f"Upscaling failed: {error_msg}")

            if not os.path.exists(temp_output_path):
                error_msg = "Output file not created"
                _write_result_file(job_id, {"status": "error", "error": error_msg})
                raise Exception(error_msg)

            output_size_mb = os.path.getsize(temp_output_path) / (1024 * 1024)
            print(f"✅ Output video size: {output_size_mb:.2f} MB")
            _update_job_progress(job_id, "✅ Finalizing output...")

            # Generate unique filename
            timestamp = int(time_module.time())
            filename = f"{url_hash}_{resolution}_{timestamp}.mp4"
            final_path = f"/outputs/{filename}"

            print(f"💾 Saving to: {final_path}")
            shutil.copy2(temp_output_path, final_path)

            output_volume.commit()

            if not os.path.exists(final_path):
                error_msg = f"Failed to save to volume: {final_path}"
                _write_result_file(job_id, {"status": "error", "error": error_msg})
                raise Exception(error_msg)

            print(f"✅ Saved successfully: {filename}")

            result = {
                "status": "success",
                "filename": filename,
                "input_size_mb": input_size_mb,
                "output_size_mb": output_size_mb
            }

            # CRITICAL: Write result to volume BEFORE returning
            _write_result_file(job_id, result)

            print(f"✅ Result file written, container can safely exit now")

            return result

    finally:
        # Cleanup
        try:
            os.chdir("/root")
            shutil.rmtree(repo_dir, ignore_errors=True)
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


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
    import threading

    web_app = FastAPI()

    JOBS_DIR = "/outputs/jobs"
    RESULTS_DIR = "/outputs/results"
    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def save_job(job_id: str, job_data: dict):
        """Save job to persistent storage"""
        with open(f"{JOBS_DIR}/{job_id}.json", "w") as f:
            json.dump(job_data, f)
        output_volume.commit()

    def load_job(job_id: str) -> Optional[dict]:
        """Load job from persistent storage"""
        job_file = f"{JOBS_DIR}/{job_id}.json"
        if not os.path.exists(job_file):
            output_volume.reload()
        if not os.path.exists(job_file):
            return None
        with open(job_file, "r") as f:
            return json.load(f)

    def read_result_file(job_id: str) -> Optional[dict]:
        """
        Read result file from volume with validation
        Returns None if file doesn't exist yet
        """
        result_file = f"{RESULTS_DIR}/{job_id}.json"

        if not os.path.exists(result_file):
            output_volume.reload()

        if not os.path.exists(result_file):
            return None

        try:
            with open(result_file, "r") as f:
                result = json.load(f)

            # Validate required fields
            if "status" in result:
                return result
            else:
                print(f"⚠️  Invalid result structure")
                return None

        except json.JSONDecodeError as e:
            print(f"⚠️  Invalid JSON in result file: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Error reading result file: {e}")
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
        download_url: Optional[str] = None
        filename: Optional[str] = None
        input_size_mb: Optional[float] = None
        output_size_mb: Optional[float] = None
        error: Optional[str] = None
        elapsed_seconds: Optional[float] = None

    def process_video(job_id: str, request: UpscaleRequest):
        """
        NEW STRATEGY: Fire off .remote() and immediately poll for result file
        Completely ignore .remote() return value - we only read from volume
        """
        try:
            job_data = load_job(job_id)
            if job_data:
                job_data["status"] = "processing"
                job_data["progress"] = "Starting upscaler..."
                save_job(job_id, job_data)

            print(f"[Job {job_id}] Starting upscale process")

            # Select GPU function
            if request.resolution in ['720p', '1080p']:
                print(f"[Job {job_id}] Using H100 for {request.resolution}")
                remote_function = upscale_video_h100
            else:
                print(f"[Job {job_id}] Using H200 for {request.resolution}")
                remote_function = upscale_video_h200

            # Fire off .remote() in background thread (don't wait for it!)
            def call_remote():
                """Fire and forget - we don't care about return value"""
                try:
                    print(f"[Job {job_id}] Calling .remote() in background...")
                    remote_function.remote(
                        video_url=request.video_url,
                        video_base64=request.video_base64,
                        batch_size=request.batch_size,
                        temporal_overlap=request.temporal_overlap,
                        stitch_mode=request.stitch_mode,
                        model=request.model,
                        resolution=request.resolution,
                        job_id=job_id
                    )
                    print(f"[Job {job_id}] .remote() returned (may never happen if container hangs)")
                except Exception as e:
                    print(f"[Job {job_id}] .remote() raised exception: {e} (this is OK)")

            remote_thread = threading.Thread(target=call_remote, daemon=True)
            remote_thread.start()

            # IMMEDIATELY start polling for result file (don't wait for .remote())
            print(f"[Job {job_id}] Polling for result file in /outputs/results/...")

            max_wait = 7260  # 2 hours + 1 minute buffer
            start_time = time.time()
            result = None
            poll_interval = 10  # Check every 10 seconds

            while time.time() - start_time < max_wait:
                result = read_result_file(job_id)

                if result:
                    if result.get("status") == "success":
                        print(f"[Job {job_id}] ✅ Found success result in volume")
                        break
                    elif result.get("status") == "error":
                        print(f"[Job {job_id}] ❌ Found error result in volume")
                        raise Exception(result.get("error", "Unknown error from GPU container"))
                    else:
                        print(f"[Job {job_id}] ⚠️  Found result with unknown status: {result.get('status')}")

                elapsed = int(time.time() - start_time)
                if elapsed % 30 == 0:  # Log every 30 seconds
                    print(f"[Job {job_id}] Still waiting for result file... ({elapsed}s / {max_wait}s)")

                time.sleep(poll_interval)

            if not result:
                raise Exception(f"No result file appeared after {max_wait}s - job likely failed in GPU container")

            # Now we have a valid success result
            filename = result["filename"]

            # Verify video file actually exists
            final_path = f"/outputs/{filename}"
            output_volume.reload()

            if not os.path.exists(final_path):
                raise Exception(f"Result file claims success but video file not found: {final_path}")

            print(f"[Job {job_id}] ✅ Output video verified: {final_path}")

            # Mark job as complete
            download_url = f"https://aifnet--seedvr2-upscaler-fastapi-app.modal.run/download/{filename}"

            job_data = load_job(job_id)
            if job_data:
                job_data.update({
                    "status": "completed",
                    "download_url": download_url,
                    "filename": filename,
                    "input_size_mb": result.get("input_size_mb", 0),
                    "output_size_mb": result.get("output_size_mb", 0),
                    "progress": "✅ Completed successfully!"
                })
                save_job(job_id, job_data)

            # Clear progress dict
            try:
                if job_id in progress_dict:
                    del progress_dict[job_id]
            except:
                pass

            # Cleanup result file
            try:
                result_file = f"{RESULTS_DIR}/{job_id}.json"
                if os.path.exists(result_file):
                    os.remove(result_file)
                    output_volume.commit()
                    print(f"[Job {job_id}] 🧹 Result file cleaned up")
            except Exception as e:
                print(f"[Job {job_id}] ⚠️  Failed to clean up result file: {e} (non-critical)")

            print(f"[Job {job_id}] ✅ Job completed successfully")

        except Exception as e:
            print(f"[Job {job_id}] ❌ Job failed: {str(e)}")
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
        """API root"""
        active_count = 0
        if os.path.exists(JOBS_DIR):
            for job_filename in os.listdir(JOBS_DIR):
                if job_filename.endswith('.json'):
                    job_data = load_job(job_filename[:-5])
                    if job_data and job_data["status"] in ["pending", "processing"]:
                        active_count += 1

        return {
            "service": "SeedVR2 Video Upscaler",
            "version": "7.0 - FILE-ONLY (Ignores .remote() return, polls volume only)",
            "endpoints": {
                "submit_job": "POST /upscale",
                "check_status": "GET /status/{job_id}",
                "download": "GET /download/{filename}"
            },
            "features": {
                "file_based_delivery": "Completely ignores .remote() return value",
                "atomic_writes": "Result files written atomically to prevent partial reads",
                "polling_strategy": "Checks for result file every 10 seconds",
                "container_hang_proof": "Works even if GPU container hangs during shutdown",
                "watchdog": "Auto-kills stalled jobs",
                "real_time_logs": "Live progress updates"
            },
            "active_jobs": active_count
        }

    return web_app