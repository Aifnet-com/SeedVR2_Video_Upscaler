# Modal SeedVR2 Architecture Flow

## Complete Job Lifecycle (Submission → Completion/Timeout)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER SUBMITS JOB                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    	     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  POST /upscale                                                          │
│  {                                                                      │
│    "video_url": "https://...",                                          │
│    "resolution": "1080p",                                               │
│    "batch_size": 100                                                    │
│  }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FastAPI App (Always Running)                                           │
│  • Generate job_id = uuid4()                                            │
│  • Determine GPU: 720p/1080p → H100, 2K → H200	                  │
│  • Save job to /outputs/jobs/{job_id}.json                              │
│    {                                                                    │
│      "status": "pending",                                               │
│      "gpu_type": "H100",                                                │
│      "created_at": timestamp                                            │
│    }                                                                    │
│  • Return immediately: {"job_id": "abc-123", "gpu_type": "H100"}        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  asyncio.create_task(process_video, job_id, request)                    │
│  Background task spawned - user doesn't wait                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Modal Function Call (GPU Container Spins Up)                           │
│  • upscale_video_h100.remote() → Runs on H100 GPU                       
│  • upscale_video_h200.remote() → Runs on H200 GPU                       
│                                                                         │
│  Container boots with:                                                  │
│  • PyTorch, CUDA, ffmpeg                                                │
│  • /models volume (persistent models)                                   │
│  • /outputs volume (persistent output storage)                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  _upscale_video_impl() - Main Processing Function                       │
│                                                                         │
│  PHASE 1: INITIALIZATION                                                │
│  ├─ git clone SeedVR2 repo → /tmp/seedvr_XXXXX                         
│  ├─ Download video from URL or decode base64                            │
│  ├─ Analyze video: width, height, fps, frame_count                      │
│  └─ Calculate stall timeout based on resolution + batch_size            │
│                                                                         │
│  PHASE 2: START PROCESSING + WATCHDOG                                   │
│  ├─ subprocess.Popen(inference_cli.py) with stdout streaming            │
│  ├─ Start watchdog_thread() in background                               │
│  │   └─ Every 10s: check if (now - last_heartbeat) > timeout            │
│  │       └─ YES → kill process group + raise exception                 
│  │       └─ NO → continue monitoring                                   
│  └─ Stream logs line by line:                                           
│      ├─ Any log line → reset last_heartbeat                             
│      ├─ "Window 0-99" → update progress_dict + job file                 
│      ├─ "Time batch: 61s" → update progress                             
│      └─ Print all logs to Modal console                                 
│                                                                         
│  PHASE 3: BATCH PROCESSING (Inside inference_cli.py)                     
│  ├─ Load models (7B DiT + VAE) → ~20s                                   
│  ├─ Process Window 0-99 (100 frames) → ~62s @ 1080p                     
│  ├─ Process Window 88-187 (100 frames) → ~62s                           
│  ├─ Process Window 176-191 (16 frames) → ~11s                           
│  └─ Stitch frames together with crossfade                                │
│                                                                          │
│  PHASE 4: FINALIZATION                                                   │
│  ├─ Save output.mp4 to /tmp                                              │
│  ├─ Copy to /outputs/{hash}_{resolution}_{timestamp}.mp4                 │
│  ├─ Commit volume to persist file                                        │
│  └─ Return {"filename": "...", "input_size_mb": ..., ...}                │
└──────────────────────────────────────────────────────────────────────────┘
                                     │
                                                       ▼
        ┌────────────────────────────┴──────────────────────────┐
        │                                                       │
            ▼                                                                                 ▼
┌────────────────────┐                              ┌──────────────────────┐
│   SUCCESS PATH     │                              │   FAILURE PATHS      │
└────────────────────┘                              └──────────────────────┘
        │                                                       │
           ▼                                                                                  ▼
┌─────────────────────────────────────┐      ┌────────────────────────────┐
│  Update job status:                 │      │  A) WATCHDOG TIMEOUT       │
│  {                                  │      │  • No logs for >timeout    │
│    "status": "completed",           │      │  • watchdog kills process  │
│    "download_url": "https://...",   │      │  • GPU freed immediately   │
│    "filename": "abc_1080p_123.mp4", │      │                            │
│    "input_size_mb": 0.55,           │      │  • Mark as "failed"        │
│    "output_size_mb": 6.49           │      └────────────────────────────┘
│  }                                  │                     │
│  • Save to job file                 │                     ▼
│  • Clear from progress_dict         │      ┌────────────────────────────┐
└─────────────────────────────────────┘      │  B) PROCESS ERROR          │
        │                                    │  • inference_cli.py fails  │
        │                                    │  • returncode != 0         │
        │                                    │  • Last 20 log lines saved │
        │                                    │  • Retry (if <2 retries)   │
        │                                    │  • OR mark as "failed"     │
        │                                    └────────────────────────────┘
        │                                                     │
        │                                                     ▼
        │                                      ┌────────────────────────────┐
        │                                      │  C) MODAL TIMEOUT (7200s)  │
        │                                      │  • 2 hour hard limit       │
        │                                      │  • Container force-killed  │
        │                                      │  • GPU freed               │
        │                                      │  • Mark as "failed"        │
        │                                      └────────────────────────────┘
        │                                                     │
        └─────────────────────────┬───────────────────────────┘
                                  │
                                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  GPU Container Shutdown                                                 │
│  • Process exits (success or failure)                                   │
│  • CUDA context destroyed                                               │
│  • VRAM freed automatically                                             │
│  • Temp files cleaned up (/tmp/seedvr_XXXXX deleted)                    │
│  • Container destroyed                                                  │
│  • GPU available for next job                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## User Polling Loop

```
┌─────────────────────────────────────────────────────────────────────────┐
│  User Script: upscale.sh or Python client                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Poll Loop (every 5 seconds)                                            │
│  GET /status/{job_id}                                                   │
│                                                                         │
│  Response:                                                              │
│  {                                                                      │
│    "job_id": "abc-123",                                                 │
│    "status": "processing",                                              │
│    "progress": "🧩 Window 88-187 (len=100)",  ← Real-time!             
│    "elapsed_seconds": 125.3                                             │
│  }                                                                      │
│                                                                         │
│  Progress comes from:                                                   │
│  1. progress_dict (in-memory, FAST) ← Updated every log                
│  2. Job file (persistent) ← Updated periodically                       
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                                      ▼
                        ┌───────────┴───────────┐
                        │                       │
                        	▼                                 ▼
            ┌──────────────────┐    ┌─────────────────────┐
            │ status="pending" │    │ status="processing"  
            │ Keep polling...  │    │ Show progress...     
            └──────────────────┘    └─────────────────────┘
                        │                       │
                        └───────────┬───────────┘
                                    │
                                    	      ▼
                        ┌───────────┴───────────┐
                        │                       │
                        	▼                                 ▼
            ┌───────────────────┐    ┌─────────────────────┐
            │ status="completed"│    │ status="failed"     │
            │ GET /download/... │    │ Show error message  │
            │ Save to disk      │    │ Exit with error     │
            └───────────────────┘    └─────────────────────┘
```

---

## Architecture Components

### 1. **FastAPI App** (Always Running, No GPU)
- **Purpose**: Job queue manager, status API
- **Location**: Modal ASGI function
- **Storage**: `/outputs/jobs/{job_id}.json`
- **Scaling**: Single instance, lightweight

### 2. **GPU Functions** (On-Demand, Short-Lived)
- **Purpose**: Heavy computation (video upscaling)
- **Types**: 
  - `upscale_video_h100()` - 80GB VRAM
  - `upscale_video_h200()` - 141GB VRAM
  - `upscale_video_b200()` - 192GB VRAM
- **Lifecycle**: Boot → Process → Shutdown
- **Scaling**: Up to 10 concurrent containers per GPU type

### 3. **Persistent Volumes**
- **Model Volume** (`/models`):
  - Stores: DiT models, VAE weights, embeddings
  - Shared across all GPU containers
  - ~20GB total
  
- **Output Volume** (`/outputs`):
  - Stores: Final videos, job metadata
  - Shared across FastAPI + GPU containers
  - Grows with usage

### 4. **In-Memory Dict** (`progress_dict`)
- **Purpose**: Ultra-fast real-time progress updates
- **Shared**: Across all Modal functions
- **Lifetime**: Survives container restarts
- **Cleared**: After job completes/fails

### 5. **Watchdog System**
- **Purpose**: Detect and kill stalled jobs
- **Mechanism**: Background thread monitoring heartbeat
- **Timeout**: Dynamic based on resolution + batch size
  - 1080p: ~3-5 min per batch
  - 2K: ~5-8 min per batch
  - 4K: ~8-12 min per batch
- **Action**: `killpg()` → GPU freed immediately

---

## Retry Logic

```
Attempt 1 (Initial)
    ├─ Success → Done ✅
    └─ Watchdog Timeout → Attempt 2

Attempt 2 (Retry 1)
    ├─ Success → Done ✅
    └─ Watchdog Timeout → Attempt 3

Attempt 3 (Retry 2 - Final)
    ├─ Success → Done ✅
    └─ Any Failure → Mark as Failed ❌
```

**Retry Conditions:**
- ✅ Retries on: Watchdog timeout, transient errors
- ❌ No retry on: Process errors (bad video, OOM), manual kills

---

## Data Flow

```
Video URL/Base64
       ↓
   Download to /tmp
       ↓
   Analyze dimensions
       ↓
   Clone SeedVR2 repo
       ↓
   Run inference_cli.py ──→ [Watchdog monitors]
       ↓                           ↓
   Extract frames            No logs for >timeout?
       ↓                           ↓
   Load models (from /models)     Kill process
       ↓                           ↓
   Process batches            GPU freed
       ↓
   Stitch frames
       ↓
   Encode to MP4
       ↓
   Save to /outputs ─────→ [User downloads]
       ↓
   Update job status
       ↓
   Clean up /tmp
       ↓
   Container shutdown
       ↓
   GPU freed
```

---

## Timeout Hierarchy

```
Level 1: Watchdog (Batch-Level)
├─ 1080p batch: ~3-5 min
├─ 2K batch: ~5-8 min
└─ 4K batch: ~8-12 min
    ↓
    If stalled: Kill process + Retry

Level 2: Modal Timeout (Job-Level)
├─ Hard limit: 7200s (2 hours)
└─ If exceeded: Force kill + No retry

Level 3: Network Timeout
├─ Download video: 300s (5 min)
└─ If exceeded: Job fails immediately
```

---

## Resource Management

| Resource        | Lifecycle                  | Cleanup              |
|-----------------|----------------------------|----------------------|
| GPU Container   | Per job                    | Auto (on exit)       |
| VRAM            | During processing          | Auto (on exit)       |
| Temp files      | Per job in /tmp            | Manual (shutil.rmtree) |
| Model cache     | Persistent (/models)       | Never (shared)       |
| Output files    | Persistent (/outputs)      | Manual (user deletes)|
| Progress dict   | Per job (in-memory)        | Manual (del dict[id])|
| Job metadata    | Persistent (JSON files)    | Manual (user deletes)|

---

## Summary

**Key Strengths:**
✅ Dynamic GPU selection (H100/H200/B200)
✅ Real-time progress updates
✅ Automatic stall detection and recovery
✅ Retry logic for transient failures
✅ Persistent storage for models and outputs
✅ Automatic GPU cleanup

**Key Constraints:**
⚠️ 2-hour hard timeout per job
⚠️ Max 10 concurrent containers per GPU type
⚠️ Network access limited to approved domains
⚠️ No cross-container communication

**Flow Duration Examples:**
- 8-sec 1080p video: ~4 min total
- 8-sec 2K video: ~8 min total
