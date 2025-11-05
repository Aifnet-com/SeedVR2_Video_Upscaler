sequenceDiagram
    autonumber
    actor U as User
    participant C as Client (upscale.sh)
    participant API as FastAPI (/upscale, /status)
    participant W as Background Worker (process_video)
    participant H as Modal GPU (H100/H200)
    participant P as progress_dict (in-memory)
    participant J as /outputs/jobs/{job_id}.json
    participant CDN as BunnyCDN (Storage + CDN URL)

    %% Submit
    U->>C: Run upscale.sh (video_url, resolution)
    C->>API: POST /upscale { video_url, resolution, batch_size }
    API->>API: Generate job_id, choose GPU (H100/H200)
    API->>J: Save {status: "pending", gpu_type, created_at}
    API-->>C: { job_id, gpu_type, status: "pending" }
    API->>W: spawn process_video(job_id, request)

    %% Status polling
    loop Poll every ~5s
        C->>API: GET /status/{job_id}
        API->>P: read realtime progress (if any)
        API->>J: read persisted job record (if any)
        API-->>C: {status, progress, download_url?}
    end

    %% Processing
    W->>H: upscale_video_{h100|h200}.remote()
    H->>H: Clone repo, download/parse video, calc stall timeout
    par Streaming logs
        H-->>P: progress updates (Window/Batch/Time logs)
        H-->>J: persisted progress (best effort)
        H-->>API: (readable via /status)
    and Watchdog
        H->>H: watchdog: kill if no logs > timeout
    end
    H->>H: inference_cli.py (process windows, stitch)

    %% Finalization
    H->>CDN: Upload output.mp4 (HTTP PUT)
    H-->>W: { filename, cdn_url, input_size_mb, output_size_mb }

    %% Complete
    W->>J: Save { status: "completed", download_url: cdn_url, ... }
    W->>P: delete progress_dict[job_id]
    API-->>C: { status: "completed", download_url }

    %% Failures
    opt Failure paths
        note over H: Watchdog timeout OR non-zero return OR 7200s hard limit
        H-->>W: raise Exception(...) (with tail logs)
        W->>J: Save { status: "failed", error, progress }
        W->>P: delete progress_dict[job_id]
        API-->>C: { status: "failed", error }
    end

