#!/bin/bash

# SeedVR2 Video Upscaler - Simple CLI wrapper
# Usage: ./upscale.sh "https://example.com/video.mp4" [--resolution 720p|1080p]

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

# Generate output filename
SHORT_ID=${JOB_ID:0:8}
OUTPUT_FILE="upscaled_${SHORT_ID}.mp4"
echo "📁 Output will be saved to: $OUTPUT_FILE"
echo ""

# Poll status
while true; do
    STATUS=$(curl -s "$API_URL/status/$JOB_ID")
    STATE=$(echo $STATUS | jq -r '.status' 2>/dev/null)

    # Handle initial sync delay (404s)
    if [ -z "$STATE" ] || [ "$STATE" = "null" ]; then
        printf "\r\033[K⏳ Waiting for job to be registered..."
        sleep 2
        continue
    fi

    ELAPSED=$(echo $STATUS | jq -r '.elapsed_seconds')
    PROGRESS=$(echo $STATUS | jq -r '.progress')

    if [ "$STATE" = "completed" ]; then
        # Clear the progress line and move to new line
        printf "\r\033[K"
        echo ""
        echo "✅ Job completed!"
        DOWNLOAD_URL=$(echo $STATUS | jq -r '.download_url')
        OUTPUT_SIZE=$(echo $STATUS | jq -r '.output_size_mb')
        echo "📊 Output: ${OUTPUT_SIZE} MB"
        echo ""

        echo "📥 Downloading..."
        curl -s "$DOWNLOAD_URL" -o "$OUTPUT_FILE"

        if [ -f "$OUTPUT_FILE" ]; then
            FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
            echo "✅ Saved to: $OUTPUT_FILE ($FILE_SIZE)"
            echo "⏱️  Total time: ${ELAPSED} seconds"
        else
            echo "❌ Download failed"
            exit 1
        fi
        break

    elif [ "$STATE" = "failed" ]; then
        # Clear the progress line
        printf "\r\033[K"
        ERROR=$(echo $STATUS | jq -r '.error')
        echo "❌ Job failed: $ERROR"
        exit 1

    else
        MINS=$((${ELAPSED%.*} / 60))
        SECS=$((${ELAPSED%.*} % 60))
        
        # Clear line, then print status (all in one printf to avoid flicker)
        printf "\r\033[K⏳ Status: $STATE [$PROGRESS] - Elapsed: ${MINS}m ${SECS}s"
    fi

    sleep 5
done
