# Seedvr2 video upscaling on Modal

Deploying on modal:
modal deploy modal_webhook.py

Submitting a job:
git clone https://github.com/Aifnet-com/SeedVR2_Video_Upscaler.git
cd SeedVR2_Video_Upscaler
bash upscale.sh "video_url" --resolution 720p/1080p/2k

Test command:

bash upscale.sh "https://aifnet.b-cdn.net/tests/test_video_upscaler/dollars_first_8sec/gladi_3s.mp4" --resolution 1080p


## 1. User Submits Job

The user sends a POST /upscale request:

{
  "video_url": "https://...",
  "resolution": "1080p",
  "batch_size": 100
}


The FastAPI service responds immediately with:

{
  "job_id": "abc-123",
  "gpu_type": "H100",
  "status": "pending"
}


## 2. FastAPI Service

Generates a unique job_id.

Determines GPU type:

720p / 1080p → H100

2K → H200

Saves the job metadata to /outputs/jobs/{job_id}.json.

Spawns a background async task:

asyncio.create_task(process_video(job_id, request))



## 3. Background Worker (process_video)

Launches a Modal GPU container by calling one of:

upscale_video_h100.remote()

upscale_video_h200.remote()

Each runs in a prebuilt Modal image with:

PyTorch, CUDA, ffmpeg

/models volume (shared weights)

/outputs volume (temporary output space)

The user can immediately poll GET /status/{job_id}.



## 4. GPU Container Workflow (_upscale_video_impl)

Phase 1: Initialization

Clones the SeedVR2 repo into /tmp.

Downloads or decodes the input video.

Extracts metadata (resolution, FPS, frame count).

Calculates the expected stall timeout based on resolution and batch size.

Phase 2: Processing + Watchdog

Runs inference_cli.py via subprocess.Popen.

Starts a watchdog thread that:

Checks log activity every 10s.

Kills the process if no logs appear for longer than the timeout.

Streams logs in real time to:

Modal console.

progress_dict (for live status updates).

/outputs/jobs/{job_id}.json (for persistence).

Phase 3: Batch Processing

Loads the diffusion + VAE models (~20s).

Processes sequential frame batches:

Example at 1080p: three 100-frame windows (~60s each).

Stitches frames together (crossfade).

Phase 4: Finalization

Saves the result as output.mp4.

Uploads directly to BunnyCDN Storage.

Returns:

{
  "filename": "abc_1080p_123.mp4",
  "cdn_url": "https://aifnet.b-cdn.net/tests/seedvr2_results/abc_1080p_123.mp4",
  "input_size_mb": 0.55,
  "output_size_mb": 6.49
}



## 5. Status Tracking
Success Path

Job JSON updated to:

{
  "status": "completed",
  "download_url": "https://aifnet.b-cdn.net/tests/seedvr2_results/abc_1080p_123.mp4"
}


Entry removed from progress_dict.

Failure Paths
Failure Type	Trigger	Result
Watchdog Timeout	No logs > stall timeout	Process killed, marked as failed
Process Error	Non-zero return code	Captures last 20 log lines, marked failed
Modal Timeout	>7200s job limit	Container killed, job failed



## 6. GPU Container Shutdown

Process exits (success or failure).

CUDA context destroyed, VRAM freed.

/tmp directories cleaned up.

Container destroyed, GPU becomes available for next job.

## 🚀 7. Backup Logic (Fail-Safe Completion)

When a job’s logs stop updating, a **fallback thread** ensures completion by checking if the output appears on BunnyCDN.

| Step | Description |
|:---- |:------------|
| 🏁 Job start | Main upscaling begins + fallback starts in parallel |
| ⏱ ETA wait | Waits for `estimate_eta_seconds()` duration |
| 🔍 CDN check | Looks for `<video_name>_<res>.mp4` on BunnyCDN |
| ✅ File found | Marks job as **completed** instantly |
| 🔁 Missing file | Retries up to 3× (every 10 seconds) |
| ❌ Timeout | Marks job as **failed** (`output file not found`) |
| 🧠 Normal path | Regular completion overrides fallback |

> 💡 **Why:** Prevents jobs from getting “stuck” if the main worker hangs.
