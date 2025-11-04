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
    import shutil
    import hashlib
    import requests
    from pathlib import Path

    def calculate_file_size_mb(filepath: str) -> float:
        """Calculate file size in MB"""
        return os.path.getsize(filepath) / (1024 * 1024)

    def log_progress(message: str):
        """Helper to log progress"""
        print(f"[{gpu_type}] {message}")
        _update_job_progress(job_id, message)

    try:
        log_progress("🚀 Starting upscaler...")

        # Download model if needed
        MODEL_DIR = "/models"
        model_path = f"{MODEL_DIR}/{model}"

        if not os.path.exists(model_path):
            log_progress(f"📥 Downloading model: {model}")
            os.makedirs(MODEL_DIR, exist_ok=True)

            model_url = f"https://huggingface.co/Seedvr/SeedVR2-Diffusion/resolve/main/{model}?download=true"
            response = requests.get(model_url, stream=True)
            response.raise_for_status()

            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            model_volume.commit()
            log_progress(f"✅ Model downloaded: {model}")
        else:
            log_progress(f"✅ Using cached model: {model}")

        # Prepare input video
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = f"{temp_dir}/input.mp4"

            if video_url:
                log_progress(f"📥 Downloading video from URL...")
                response = requests.get(video_url, stream=True)
                response.raise_for_status()
                with open(input_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            elif video_base64:
                log_progress(f"📥 Decoding base64 video...")
                video_bytes = b64decode(video_base64)
                with open(input_path, 'wb') as f:
                    f.write(video_bytes)
            else:
                raise ValueError("Must provide either video_url or video_base64")

            input_size_mb = calculate_file_size_mb(input_path)
            log_progress(f"📊 Input video: {input_size_mb:.2f} MB")

            # Resolution mapping
            resolution_map = {
                '720p': (1280, 720),
                '1080p': (1920, 1080),
                '2k': (2560, 1440),
                '4k': (3840, 2160)
            }

            target_width, target_height = resolution_map.get(resolution, (1920, 1080))
            log_progress(f"🎯 Target resolution: {resolution} ({target_width}x{target_height})")

            # Run upscaling with watchdog
            output_path = f"{temp_dir}/upscaled.mp4"
            log_progress(f"🎬 Starting upscale process...")

            stall_timeout = _calculate_stall_timeout(resolution, batch_size)

            cmd = [
                "python", "-u", "/root/ComfyUI-SeedVR2_VideoUpscaler/seedvr2_upscaler.py",
                "--input", input_path,
                "--output", output_path,
                "--model_path", model_path,
                "--batch_size", str(batch_size),
                "--temporal_overlap", str(temporal_overlap),
                "--stitch_mode", stitch_mode,
                "--target_width", str(target_width),
                "--target_height", str(target_height)
            ]

            log_progress(f"⚙️  Processing with batch_size={batch_size}, overlap={temporal_overlap}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            last_output_time = time.time()
            last_progress = ""

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                if line:
                    line = line.strip()
                    if line:
                        print(line)
                        last_output_time = time.time()

                        # Extract and report progress
                        if "Processing batch" in line or "frames processed" in line or "Upscaling" in line:
                            if line != last_progress:
                                log_progress(line)
                                last_progress = line

                # Check for stall
                elapsed_since_output = time.time() - last_output_time
                if elapsed_since_output > stall_timeout:
                    log_progress(f"⚠️  Process stalled (no output for {stall_timeout}s), killing...")
                    process.kill()
                    process.wait()
                    raise Exception(f"Upscaling process stalled for {stall_timeout}s")

            return_code = process.wait()

            if return_code != 0:
                raise Exception(f"Upscaling failed with code {return_code}")

            if not os.path.exists(output_path):
                raise Exception("Output video file was not created")

            output_size_mb = calculate_file_size_mb(output_path)
            log_progress(f"✅ Upscaling complete! Output: {output_size_mb:.2f} MB")

            # Generate unique filename and save to volume
            file_hash = hashlib.md5(f"{job_id or 'no-job'}_{time.time()}".encode()).hexdigest()[:8]
            timestamp = int(time.time())
            final_filename = f"{file_hash}_{resolution}_{timestamp}.mp4"
            final_path = f"/outputs/{final_filename}"

            log_progress(f"💾 Saving to volume: {final_filename}")
            shutil.copy2(output_path, final_path)
            output_volume.commit()

            log_progress(f"✅ Video saved successfully!")

            # Write success result to volume
            result_data = {
                "status": "success",
                "filename": final_filename,
                "input_size_mb": input_size_mb,
                "output_size_mb": output_size_mb,
                "resolution": resolution,
                "gpu_type": gpu_type
            }

            _write_result_file(job_id, result_data)

            return result_data

    except Exception as e:
        error_msg = str(e)
        log_progress(f"❌ Error: {error_msg}")

        # Write error result to volume
        result_data = {
            "status": "error",
            "error": error_msg,
            "gpu_type": gpu_type
        }

        _write_result_file(job_id, result_data)

        raise


@app.function(
    image=image,
    scaledown_window=300,
    volumes={"/outputs": output_volume},
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI webhook for job submission and status checking"""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    import uuid
    import asyncio

    web_app = FastAPI()

    JOBS_DIR = "/outputs/jobs"
    RESULTS_DIR = "/outputs/results"
    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def save_job(job_id: str, job_data: dict):
        """Save job to persistent storage with error handling"""
        try:
            job_file = f"{JOBS_DIR}/{job_id}.json"
            temp_file = f"{JOBS_DIR}/{job_id}.tmp"

            # Write to temp file first
            with open(temp_file, "w") as f:
                json.dump(job_data, f)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            os.rename(temp_file, job_file)
            output_volume.commit()
        except Exception as e:
            print(f"⚠️  Error saving job {job_id}: {e}")
            # Try cleanup
            try:
                if os.path.exists(f"{JOBS_DIR}/{job_id}.tmp"):
                    os.remove(f"{JOBS_DIR}/{job_id}.tmp")
            except:
                pass

    def load_job(job_id: str) -> Optional[dict]:
        """Load job from persistent storage with error handling"""
        job_file = f"{JOBS_DIR}/{job_id}.json"

        try:
            if not os.path.exists(job_file):
                output_volume.reload()

            if not os.path.exists(job_file):
                return None

            with open(job_file, "r") as f:
                return json.load(f)

        except (IOError, OSError) as e:
            # File might be locked by another container
            print(f"⚠️  Could not read job {job_id}: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"⚠️  Invalid JSON in job file {job_id}: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Unexpected error loading job {job_id}: {e}")
            return None

    def read_result_file(job_id: str) -> Optional[dict]:
        """
        Read result file from volume with validation and error handling
        Returns None if file doesn't exist yet or is locked
        """
        result_file = f"{RESULTS_DIR}/{job_id}.json"

        try:
            if not os.path.exists(result_file):
                output_volume.reload()

            if not os.path.exists(result_file):
                return None

            with open(result_file, "r") as f:
                result = json.load(f)

            # Validate required fields
            if "status" in result:
                return result
            else:
                print(f"⚠️  Invalid result structure for {job_id}")
                return None

        except (IOError, OSError) as e:
            # File might be locked by another container
            print(f"⚠️  Could not read result file {job_id}: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"⚠️  Invalid JSON in result file {job_id}: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Unexpected error reading result {job_id}: {e}")
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
        """API root - returns static info without reading job files"""
        return {
            "service": "SeedVR2 Video Upscaler",
            "version": "8.0 - FILE-ONLY with Concurrency Fix",
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
                "real_time_logs": "Live progress updates",
                "concurrency_safe": "Jobs isolated by ID, no cross-job file access"
            },
            "note": "Active job count removed to prevent file locking issues"
        }

    return web_app