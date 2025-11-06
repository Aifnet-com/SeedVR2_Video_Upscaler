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
                resp = requests.put(url, headers=headers, data=f, timeout=600)
            if resp.status_code in (200, 201):
                return f"{base_url}/{zone_rel_path}"
            else:
                print(f"⚠️ Bunny upload failed [{resp.status_code}]: {resp.text[:300]}")
        except Exception as e:
            print(f"⚠️ Bunny upload error (attempt {attempt+1}/3): {e}")
        _t.sleep(3 * (attempt + 1))

    raise Exception("Bunny upload failed after 3 attempts")

# ---------------- Fallback helpers (CDN probing) ----------------

def derive_output_paths(video_url: str, resolution: str):
    """Return (filename, zone_rel_path, cdn_url) based on input URL + resolution."""
    import os, urllib.parse
    _, _, _, base_url, root_dir = _bunny_cfg()
    parsed = urllib.parse.urlparse(video_url or "")
    base = os.path.basename(parsed.path) or "video.mp4"
    name_root, _ = os.path.splitext(base)
    filename = f"{name_root}_{resolution}.mp4"
    zone_rel_path = f"{root_dir}/{filename}"
    cdn_url = f"{base_url}/{zone_rel_path}"
    return filename, zone_rel_path, cdn_url

def probe_video_meta(url: str):
    """Use ffprobe on the remote input URL to estimate duration/fps/frames (no full download)."""
    import subprocess, json as _json
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate:format=duration",
                "-of", "json",
                url,
            ],
            text=True,
            timeout=20,
        )
        data = _json.loads(out)
        dur = float(data["format"]["duration"])
        fr = data["streams"][0]["avg_frame_rate"]  # e.g., "25/1"
        n, d = fr.split("/")
        fps = float(n) / float(d or 1)
        frames = max(1, int(round(dur * fps)))
        return {"duration": dur, "fps": fps, "frames": frames}
    except Exception:
        return None


def estimate_eta_seconds(frames: int, resolution: str, batch_size: int = 100):
    per100_gpu = {"720p": 60, "1080p": 65, "2k": 120, "4k": 250}
    base_gpu = per100_gpu.get(resolution, 65)
    num_batches = max(1, int((frames + batch_size - 1) / batch_size))

    gpu_time = base_gpu * (frames / 100.0)
    model_reload_time = 45 + (15 * max(0, num_batches - 1))
    stitch_time = 2 * max(0, num_batches - 1)
    save_encode_time = (3.5 * (frames / 100.0)) + 10
    upload_time = max(5, int(frames / 100.0 * 2))

    total = gpu_time + model_reload_time + stitch_time + save_encode_time + upload_time
    return int(total * 1.25)  # 25% safety margin

def cdn_file_exists(cdn_url: str):
    """HEAD the CDN URL; return (exists, size_bytes)."""
    import requests
    try:
        r = requests.head(cdn_url, timeout=10)
        if r.status_code == 200:
            size = int(r.headers.get("Content-Length", "0") or 0)
            return True, size
    except Exception:
        pass
    return False, 0

# ---------------- Utility: progress + timeout heuristics ----------------

def _update_job_progress(job_id: str, progress_text: str):
    """Update progress in shared dict (fast path)."""
    if not job_id:
        return
    import time
    try:
        progress_dict[job_id] = {"text": progress_text, "timestamp": time.time()}
    except Exception as e:
        print(f"⚠️ progress_dict update failed: {e}")

def _calculate_stall_timeout(resolution: str, batch_size: int = 100) -> int:
    """
    Timeout must exceed per-batch processing time (we only log between batches).
    Empirical seconds per 100 frames:
      720p ~50, 1080p ~70, 2k ~120, 4k ~250.
    """
    time_per_100_frames = {"720p": 50, "1080p": 70, "2k": 120, "4k": 250}
    base = time_per_100_frames.get(resolution, 62)
    expected_batch = int(base * (batch_size / 100))
    first_batch_timeout = max(int((expected_batch + 45) * 1.5), 180)  # add model load + slack
    return first_batch_timeout

# ---------------- Core upscaler implementation ----------------

def _upscale_video_impl(
    video_url: Optional[str] = None,
    video_base64: Optional[str] = None,
    batch_size: int = 100,
    temporal_overlap: int = 12,
    stitch_mode: str = "crossfade",
    model: str = "seedvr2_ema_7b_fp16.safetensors",
    resolution: str = "1080p",
    gpu_type: str = "H100",
    job_id: Optional[str] = None,
):
    """
    Run SeedVR2, then upload the resulting MP4 to Bunny Storage.
    Returns filename, sizes, and the direct CDN URL (cdn_url).
    """
    import subprocess, tempfile, os, requests, hashlib, time as time_module
    import shutil, cv2, math, threading, signal
    import urllib.parse

    print(f"🚀 Starting SeedVR2 on {gpu_type} @ {resolution}")
    _update_job_progress(job_id, "🚀 Initializing upscaler...")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

    repo_dir = tempfile.mkdtemp(prefix="seedvr_")
    _update_job_progress(job_id, "📂 Cloning repository...")
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/Aifnet-com/SeedVR2_Video_Upscaler.git", repo_dir],
            check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Git clone failed: {e.stderr}")
        raise
    os.chdir(repo_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mp4")
        output_tmp = os.path.join(tmpdir, "output.mp4")

        # Input fetch/prepare
        if video_url:
            _update_job_progress(job_id, "📥 Downloading video...")
            r = requests.get(video_url, stream=True, timeout=300)
            r.raise_for_status()
            with open(input_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            url_hash = hashlib.md5(video_url.encode()).hexdigest()[:8]
        elif video_base64:
            _update_job_progress(job_id, "📥 Decoding video...")
            from base64 import b64decode as _b64
            with open(input_path, "wb") as f:
                f.write(_b64(video_base64))
            url_hash = hashlib.md5(video_base64.encode()).hexdigest()[:8]
        else:
            raise Exception("Must provide either video_url or video_base64")

        input_size_mb = os.path.getsize(input_path) / (1024 * 1024)

        # Probe
        _update_job_progress(job_id, "📐 Analyzing video...")
        cap = cv2.VideoCapture(input_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if w == 0 or h == 0:
            raise Exception(f"Could not read input dimensions ({w}x{h})")

        # Timeouts
        stall_timeout = _calculate_stall_timeout(resolution, batch_size)

        # Target dims by pixel budget (rounded to /16)
        target_pixels_map = {"720p": 921600, "1080p": 2073600, "2k": 3686400, "4k": 8294400}
        tgt = target_pixels_map.get(resolution, 2073600)
        ratio = math.sqrt(tgt / (w * h))
        out_w = round((w * ratio) / 16) * 16
        out_h = round((h * ratio) / 16) * 16
        res_px = min(out_w, out_h)

        # Build CLI
        cmd = [
            "python", "inference_cli.py",
            "--video_path", input_path,
            "--batch_size", str(batch_size),
            "--temporal_overlap", str(temporal_overlap),
            "--stitch_mode", stitch_mode,
            "--model", model,
            "--resolution", str(res_px),
            "--model_dir", "/models",
            "--output", output_tmp,
            "--debug",
        ]

        _update_job_progress(job_id, "🔧 Starting upscale process...")

        # Run with streaming + watchdog
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=repo_dir,
            preexec_fn=os.setsid
        )

        lines = []
        last_heartbeat = time_module.time()
        stalled_kill = False
        is_first_batch = True

        def watchdog():
            nonlocal stalled_kill, last_heartbeat, is_first_batch
            while proc.poll() is None:
                time_module.sleep(10)
                if time_module.time() - last_heartbeat > stall_timeout:
                    print("🚨 WATCHDOG: stalled, killing process group")
                    stalled_kill = True
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        time_module.sleep(2)
                        if proc.poll() is None:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception as e:
                        print(f"⚠️ watchdog kill error: {e}")
                    break

        import threading as _t
        _t.Thread(target=watchdog, daemon=True).start()

        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            line = line.rstrip()
            lines.append(line)
            print(line)  # modal logs
            # heuristics for progress
            p = None
            if "Window" in line and "-" in line:
                p = line.strip()
            elif "Time batch:" in line:
                p = line.strip()
                if is_first_batch:
                    is_first_batch = False
            elif "Batch" in line and "/" in line:
                p = line.strip()
            elif "Loading model" in line or "Model loaded" in line:
                p = "⚙️ " + line.strip()
            elif "Downloading" in line:
                p = "⬇️ " + line.strip()
            elif "Processing" in line and "frames" in line:
                p = "🎬 " + line.strip()

            if p:
                _update_job_progress(job_id, p)
                last_heartbeat = time_module.time()
            elif line.strip():
                last_heartbeat = time_module.time()

        proc.wait()
        if stalled_kill:
            raise Exception(f"Job stalled (>{stall_timeout}s) and was killed")
        if proc.returncode != 0:
            tail = "\n".join(lines[-20:])
            raise Exception(f"Upscaling failed:\n{tail}")
        if not os.path.exists(output_tmp):
            raise Exception("Output file not created")

        # ------------------------------------------------------------------
        # ✅ Re-encode with H.264 for optimal quality & web playback
        # ------------------------------------------------------------------
        _update_job_progress(job_id, "🎞️ Re-encoding output with libx264 (CRF 18)...")
        reencoded_path = os.path.join(tmpdir, "output_final.mp4")

        cmd = [
            "ffmpeg", "-y", "-nostdin",
            "-hide_banner", "-loglevel", "info",
            "-i", output_tmp,
            "-c:v", "libx264",
            "-profile:v", "high",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-preset", "medium",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            reencoded_path
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(proc.stdout.readline, ''):
            if line.strip():
                print("🎬 [ffmpeg]", line.strip())
        proc.wait(timeout=60)

        if proc.returncode != 0 or not os.path.exists(reencoded_path):
            raise Exception("❌ ffmpeg re-encode failed or produced no output")

        print("✅ Re-encoded successfully with H.264 + faststart", flush=True)

        # ------------------------------------------------------------------
        # 🎧 Merge original audio (if exists) back into the upscaled video
        # ------------------------------------------------------------------
        _update_job_progress(job_id, "🎧 Restoring original audio track...")
        audio_path = os.path.join(tmpdir, "audio.m4a")
        merged_path = os.path.join(tmpdir, "output_final_audio.mp4")

        # Extract audio safely
        extract_cmd = [
            "ffmpeg", "-y", "-nostdin", "-hide_banner",
            "-i", input_path, "-vn", "-acodec", "copy", audio_path
        ]
        subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Merge only if audio was extracted successfully
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1024:
            merge_cmd = [
                "ffmpeg", "-y", "-nostdin", "-hide_banner",
                "-i", reencoded_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                merged_path
            ]
            proc = subprocess.Popen(merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in iter(proc.stdout.readline, ''):
                if line.strip():
                    print("🎬 [audio merge]", line.strip())
            proc.wait(timeout=300)
            if proc.returncode != 0 or not os.path.exists(merged_path):
                print("⚠️ Audio merge failed, keeping video-only output")
            else:
                print("✅ Audio merged successfully", flush=True)
                output_tmp = merged_path
        else:
            print("ℹ️ No audio track found or extraction failed, skipping merge")
        # ------------------------------------------------------------------

        output_size_mb = os.path.getsize(output_tmp) / (1024 * 1024)
        _update_job_progress(job_id, "✅ Uploading to Bunny Storage...")

        # -------- Upload to Bunny Storage (DIRECT CDN URL) --------
        # Derive clean base name from original URL (or fallback to hash)
        parsed_url = urllib.parse.urlparse(video_url)
        base_name = os.path.basename(parsed_url.path) or f"video_{url_hash}.mp4"
        name_root, _ = os.path.splitext(base_name)

        # Final filename: <original>_<resolution>.mp4
        _, _, _, _BUNNY_BASE_URL, BUNNY_ROOT_DIR = _bunny_cfg()
        filename = f"{name_root}_{resolution}.mp4"
        zone_rel_path = f"{BUNNY_ROOT_DIR}/{filename}"
        cdn_url = upload_to_bunny_storage(output_tmp, zone_rel_path)
        print(f"✅ Uploaded to Bunny Storage: {cdn_url}")

    # Cleanup repo
    os.chdir("/root")
    import shutil as _sh
    _sh.rmtree(repo_dir, ignore_errors=True)

    return {
        "filename": filename,
        "input_size_mb": input_size_mb,
        "output_size_mb": output_size_mb,
        "cdn_url": cdn_url,   # direct file URL
    }


# ---------------- GPU wrappers ----------------

@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    secrets=[bunny_secret],
    volumes={"/models": model_volume, "/outputs": output_volume},
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
    job_id: Optional[str] = None,
):
    return _upscale_video_impl(
        video_url, video_base64, batch_size, temporal_overlap,
        stitch_mode, model, resolution, gpu_type="H100", job_id=job_id
    )

@app.function(
    image=image,
    gpu="H200",
    timeout=7200,
    secrets=[bunny_secret],
    volumes={"/models": model_volume, "/outputs": output_volume},
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
    job_id: Optional[str] = None,
):
    return _upscale_video_impl(
        video_url, video_base64, batch_size, temporal_overlap,
        stitch_mode, model, resolution, gpu_type="H200", job_id=job_id
    )

# ---------------- FastAPI app ----------------

@app.function(
    image=image,
    timeout=7200,
    secrets=[bunny_secret],
    scaledown_window=60,
    volumes={"/outputs": output_volume},
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import time, uuid, os, asyncio, json as _json
    import threading as _thread

    web_app = FastAPI()

    # Local job cache (simple; okay since one web app handles status)
    JOBS_DIR = "/outputs/jobs"
    os.makedirs(JOBS_DIR, exist_ok=True)

    def save_job(job_id: str, job_data: dict):
        try:
            with open(f"{JOBS_DIR}/{job_id}.json", "w") as f:
                _json.dump(job_data, f)
        except Exception as e:
            print(f"⚠️ save_job error: {e}")

    def load_job(job_id: str):
        try:
            path = f"{JOBS_DIR}/{job_id}.json"
            if not os.path.exists(path):
                return None
            with open(path, "r") as f:
                return _json.load(f)
        except Exception as e:
            print(f"⚠️ load_job error: {e}")
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
        download_url: Optional[str] = None  # direct CDN URL
        filename: Optional[str] = None
        input_size_mb: Optional[float] = None
        output_size_mb: Optional[float] = None
        error: Optional[str] = None
        elapsed_seconds: Optional[float] = None

    def process_video(job_id: str, request: UpscaleRequest):
        try:
            job_data = load_job(job_id) or {}
            job_data.update({"status": "processing", "progress": "Starting upscaler..."})
            save_job(job_id, job_data)

            # === Fallback completion thread (backup path if logs or updates stall) ===
            # Determine expected CDN URL upfront
            expected_filename, _expected_zone_rel, expected_cdn_url = derive_output_paths(
                request.video_url or "", request.resolution
            )

            def fallback_complete():
                import time as _t
                # Quick probe to estimate frames -> ETA
                meta = probe_video_meta(request.video_url) if request.video_url else None
                frames = (meta or {}).get("frames", 300)  # default ≈ 12s @ 25fps
                eta = max(estimate_eta_seconds(frames, request.resolution), 60)  # at least 60s

                # Wait ETA, then try up to 3 times, 10s apart
                _t.sleep(eta)
                for _ in range(3):
                    ok, size = cdn_file_exists(expected_cdn_url)
                    if ok:
                        job_data = load_job(job_id) or {}
                        job_data.update({
                            "status": "completed",
                            "progress": "✅ Completed via fallback check",
                            "download_url": expected_cdn_url,
                            "filename": filename,
                            "output_size_mb": size / (1024 * 1024) if size else None,
                        })
                        save_job(job_id, job_data)
                        print(f"✅ Fallback: Found CDN file {expected_cdn_url}, marking completed")
                        try:
                            if job_id in progress_dict:
                                del progress_dict[job_id]
                        except Exception:
                            pass
                        return
                    print(f"⚠️ Fallback: file not found yet, retrying in 10s ({_ + 1}/3)")
                    _t.sleep(10)

                # After 3 unsuccessful retries:
                job_data = load_job(job_id) or {}
                job_data.update({
                    "status": "failed",
                    "error": "❌ Fallback timeout: output file not found on BunnyCDN after retries",
                    "progress": "❌ Job failed: output file missing on CDN",
                })
                save_job(job_id, job_data)
                try:
                    if job_id in progress_dict:
                        del progress_dict[job_id]
                except Exception:
                    pass
                print("❌ Fallback: job failed - output file never appeared on CDN")

            _thread.Thread(target=fallback_complete, daemon=True).start()
            # === end fallback ===

            # Choose GPU (normal path)
            if request.resolution in ["720p", "1080p"]:
                res = upscale_video_h100.remote(
                    video_url=request.video_url,
                    video_base64=request.video_base64,
                    batch_size=request.batch_size,
                    temporal_overlap=request.temporal_overlap,
                    stitch_mode=request.stitch_mode,
                    model=request.model,
                    resolution=request.resolution,
                    job_id=job_id,
                )
            else:
                res = upscale_video_h200.remote(
                    video_url=request.video_url,
                    video_base64=request.video_base64,
                    batch_size=request.batch_size,
                    temporal_overlap=request.temporal_overlap,
                    stitch_mode=request.stitch_mode,
                    model=request.model,
                    resolution=request.resolution,
                    job_id=job_id,
                )

            # Persist results (normal completion)
            job_data = load_job(job_id) or {}
            job_data.update({
                "status": "completed",
                "download_url": res["cdn_url"],   # direct CDN file
                "filename": res["filename"],
                "input_size_mb": res["input_size_mb"],
                "output_size_mb": res["output_size_mb"],
                "progress": "✅ Completed successfully!",
            })
            save_job(job_id, job_data)

            # Clear progress entry
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
                realtime = pd.get("text") if isinstance(pd, dict) else pd
        except Exception:
            pass

        job_data = load_job(job_id)
        if not job_data:
            if realtime:
                return JobStatus(job_id=job_id, status="processing", progress=realtime)
            raise HTTPException(status_code=404, detail="Job not found")

        if realtime:
            job_data["progress"] = realtime

        created_at = job_data.get("created_at")
        elapsed = (time.time() - created_at) if created_at else None

        return JobStatus(
            job_id=job_id,
            status=job_data.get("status", "pending"),
            progress=job_data.get("progress"),
            download_url=job_data.get("download_url"),  # direct CDN file
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
            "version": "4.2 - Bunny Storage (direct URL) + CDN fallback",
            "endpoints": {
                "submit_job": "POST /upscale",
                "check_status": "GET /status/{job_id}",
            },
            "active_jobs": active,
        }

    return web_app
