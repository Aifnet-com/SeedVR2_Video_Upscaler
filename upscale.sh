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

# Track registration waiting time
REGISTRATION_START=$(date +%s)
REGISTRATION_TIMEOUT=60  # 1 minute to wait for registration
STORAGE_CHECK_ATTEMPTS=0
MAX_STORAGE_CHECKS=10
LAST_MESSAGE=""

# Poll status
while true; do
    STATUS=$(curl -s "$API_URL/status/$JOB_ID")
    STATE=$(echo $STATUS | jq -r '.status' 2>/dev/null)

    # Handle initial sync delay (404s) with timeout
    if [ -z "$STATE" ] || [ "$STATE" = "null" ]; then
        CURRENT_TIME=$(date +%s)
        ELAPSED_REGISTRATION=$((CURRENT_TIME - REGISTRATION_START))

        if [ $ELAPSED_REGISTRATION -ge $REGISTRATION_TIMEOUT ]; then
            # After 1 minute of waiting, start checking storage
            if [ $STORAGE_CHECK_ATTEMPTS -eq 0 ]; then
                echo ""
                echo "⚠️  Job not registered after 60 seconds. Checking storage for output..."
            fi

            # Check storage every 30 seconds for up to 10 times
            if [ $STORAGE_CHECK_ATTEMPTS -lt $MAX_STORAGE_CHECKS ]; then
                STORAGE_CHECK_ATTEMPTS=$((STORAGE_CHECK_ATTEMPTS + 1))
                printf "\r\033[K🔍 Checking storage... (attempt $STORAGE_CHECK_ATTEMPTS/$MAX_STORAGE_CHECKS)"

                # Wait 30 seconds between storage checks (except for the first one)
                if [ $STORAGE_CHECK_ATTEMPTS -gt 1 ]; then
                    sleep 30
                fi

                # Try to get status again after waiting
                STATUS=$(curl -s "$API_URL/status/$JOB_ID")
                STATE=$(echo $STATUS | jq -r '.status' 2>/dev/null)

                # If still no status, continue checking
                if [ -z "$STATE" ] || [ "$STATE" = "null" ]; then
                    continue
                fi
            else
                # Failed after all storage checks
                printf "\r\033[K"
                echo ""
                echo "❌ Job failed: Could not find output after $MAX_STORAGE_CHECKS storage checks"
                echo "   Job may have failed to start or output was not uploaded"
                exit 1
            fi
        else
            # Still within the 1-minute registration window
            printf "\r\033[K⏳ Waiting for job to be registered... ($ELAPSED_REGISTRATION seconds)"
            sleep 2
            continue
        fi
    fi

    # If we got here, we have a valid status
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

        # Check if it's a pending timeout failure
        if [[ "$ERROR" == *"Modal scheduling timeout"* ]]; then
            echo "   The job could not be scheduled on Modal. Please try again later."
        fi
        exit 1

    else
        # Show progress
        if [ -n "$ELAPSED" ] && [ "$ELAPSED" != "null" ]; then
            MINS=$(echo "$ELAPSED" | awk '{print int($1/60)}')
            SECS=$(echo "$ELAPSED" | awk '{print int($1%60)}')
            CURRENT_MESSAGE="⏳ Status: $STATE [$PROGRESS] - Elapsed: ${MINS}m ${SECS}s"
        else
            CURRENT_MESSAGE="⏳ Status: $STATE [$PROGRESS]"
        fi

        # Only update if message changed to reduce flickering
        if [ "$CURRENT_MESSAGE" != "$LAST_MESSAGE" ]; then
            printf "\r\033[K$CURRENT_MESSAGE"
            LAST_MESSAGE="$CURRENT_MESSAGE"
        fi
    fi

    sleep 5
done