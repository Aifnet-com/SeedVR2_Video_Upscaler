#!/bin/bash

# SeedVR2 Video Upscaler - Simple CLI wrapper
# Usage: ./upscale.sh "https://example.com/video.mp4" [--resolution 720p|1080p|2k|4k]

if [ -z "$1" ]; then
    echo "Usage: $0 <video_url> [--resolution 720p|1080p|2k|4k]"
    echo "Example: $0 'https://astra.app/api/files/xxx.mp4'"
    echo "Example: $0 'https://astra.app/api/files/xxx.mp4' --resolution 720p"
    echo "Example: $0 'https://astra.app/api/files/xxx.mp4' --resolution 4k"
    exit 1
fi

VIDEO_URL="$1"
API_URL="https://aifnet--seedvr2-upscaler-fastapi-app.modal.run"
RESOLUTION="1080p"

# Parse optional resolution argument
if [ "$2" = "--resolution" ] && [ -n "$3" ]; then
    if [[ "$3" =~ ^(720p|1080p|2k|4k)$ ]]; then
        RESOLUTION="$3"
    else
        echo "❌ Invalid resolution: $3. Must be 720p, 1080p, 2k, or 4k"
        exit 1
    fi
fi

echo "🚀 Submitting upscaling job..."

# Submit job with resolution as string
REQUEST_BODY=$(cat <<EOF
{
  "video_url": "$VIDEO_URL",
  "resolution": "$RESOLUTION"
}
EOF
)

RESPONSE=$(curl -s -X POST "$API_URL/upscale" \
  -H "Content-Type: application/json" \
  -d "$REQUEST_BODY")

JOB_ID=$(echo $RESPONSE | jq -r '.job_id' 2>/dev/null)

if [ -z "$JOB_ID" ] || [ "$JOB_ID" = "null" ]; then
    echo "❌ Failed to submit job"
    echo $RESPONSE | jq 2>/dev/null || echo $RESPONSE
    exit 1
fi

echo "✅ Job submitted: $JOB_ID"

# Extract and display GPU type
GPU_TYPE=$(echo $RESPONSE | jq -r '.gpu_type' 2>/dev/null)
if [ -n "$GPU_TYPE" ] && [ "$GPU_TYPE" != "null" ]; then
    echo "🖥️  GPU: $GPU_TYPE"
fi

echo ""

# Poll status
REGISTRATION_START=$(date +%s)
REGISTRATION_CHECKS=0

while true; do
    STATUS=$(curl -s "$API_URL/status/$JOB_ID")
    STATE=$(echo $STATUS | jq -r '.status' 2>/dev/null)

    # Handle initial sync delay (404s or null state)
    if [ -z "$STATE" ] || [ "$STATE" = "null" ]; then
        NOW=$(date +%s)
        WAIT_TIME=$((NOW - REGISTRATION_START))

        # If waiting more than 60 seconds, start checking storage
        if [ $WAIT_TIME -ge 60 ]; then
            if [ $REGISTRATION_CHECKS -lt 10 ]; then
                printf "\r\033[K⏳ Job not registered after ${WAIT_TIME}s. Checking storage (attempt $((REGISTRATION_CHECKS + 1))/10)..."
                REGISTRATION_CHECKS=$((REGISTRATION_CHECKS + 1))
                sleep 30
                continue
            else
                # Failed after 10 checks (60s + 10*30s = 360s total)
                printf "\r\033[K"
                echo ""
                echo "❌ Job failed to register after 6 minutes"
                echo "Job ID: $JOB_ID"
                echo "This likely indicates a Modal scheduling failure."
                exit 1
            fi
        else
            printf "\r\033[K⏳ Waiting for job to be registered... (${WAIT_TIME}s)"
            sleep 2
            continue
        fi
    fi

    # If we got a state, check if it's a failure (even during registration phase)
    if [ "$STATE" = "failed" ]; then
        printf "\r\033[K"
        echo ""
        ERROR=$(echo $STATUS | jq -r '.error')
        echo "❌ Job failed: $ERROR"
        exit 1
    fi

    ELAPSED=$(echo $STATUS | jq -r '.elapsed_seconds')
    PROGRESS=$(echo $STATUS | jq -r '.progress')

    if [ "$STATE" = "completed" ]; then
        printf "\r\033[K"
        echo ""
        echo "✅ Job completed!"
        DOWNLOAD_URL=$(echo $STATUS | jq -r '.download_url')
        OUTPUT_SIZE=$(echo $STATUS | jq -r '.output_size_mb')
        echo "📊 Output: ${OUTPUT_SIZE} MB"
        echo ""
        echo "Upscaled_url: ${DOWNLOAD_URL}"
        echo "⏱️  Total time: ${ELAPSED} seconds"
        break

    elif [ "$STATE" = "failed" ]; then
        printf "\r\033[K"
        ERROR=$(echo $STATUS | jq -r '.error')
        echo "❌ Job failed: $ERROR"
        exit 1

    else
        MINS=$((${ELAPSED%.*} / 60))
        SECS=$((${ELAPSED%.*} % 60))

        # Add helpful context for pending jobs
        if [ "$STATE" = "pending" ]; then
            if [ ${ELAPSED%.*} -gt 60 ]; then
                printf "\r\033[K⏳ Status: $STATE (waiting in job queue) - Elapsed: ${MINS}m ${SECS}s"
            else
                printf "\r\033[K⏳ Status: $STATE [$PROGRESS] - Elapsed: ${MINS}m ${SECS}s"
            fi
        elif [ "$STATE" = "processing" ] && [[ "$PROGRESS" == *"Starting"* ]]; then
            printf "\r\033[K⏳ Status: Waiting for GPU assignment - Elapsed: ${MINS}m ${SECS}s"
        else
            printf "\r\033[K⏳ Status: $STATE [$PROGRESS] - Elapsed: ${MINS}m ${SECS}s"
        fi
    fi

    sleep 5
done