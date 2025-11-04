#!/bin/bash

# SeedVR2 Video Upscaler - Simple CLI wrapper
# Usage: ./upscale.sh "https://example.com/video.mp4" [--resolution 720p|1080p|2k|4k] [--open]

if [ -z "$1" ]; then
    echo "Usage: $0 <video_url> [--resolution 720p|1080p|2k|4k] [--open]"
    echo "Example: $0 'https://astra.app/api/files/xxx.mp4'"
    echo "Example: $0 'https://astra.app/api/files/xxx.mp4' --resolution 720p"
    echo "Example: $0 'https://astra.app/api/files/xxx.mp4' --resolution 4k --open"
    exit 1
fi

VIDEO_URL="$1"
API_URL="https://aifnet--seedvr2-upscaler-fastapi-app.modal.run"
RESOLUTION="1080p"
AUTO_OPEN=false

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --open)
            AUTO_OPEN=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [[ ! "$RESOLUTION" =~ ^(720p|1080p|2k|4k)$ ]]; then
    echo "❌ Invalid resolution: $RESOLUTION. Must be 720p, 1080p, 2k, or 4k"
    exit 1
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

JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id' 2>/dev/null)

if [ -z "$JOB_ID" ] || [ "$JOB_ID" = "null" ]; then
    echo "❌ Failed to submit job"
    echo "$RESPONSE" | jq 2>/dev/null || echo "$RESPONSE"
    exit 1
fi

echo "✅ Job submitted: $JOB_ID"

# Extract and display GPU type
GPU_TYPE=$(echo "$RESPONSE" | jq -r '.gpu_type' 2>/dev/null)
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
    STATE=$(echo "$STATUS" | jq -r '.status' 2>/dev/null)

    # Handle initial sync delay (404s)
    if [ -z "$STATE" ] || [ "$STATE" = "null" ]; then
        printf "\r\033[K⏳ Waiting for job to be registered..."
        sleep 2
        continue
    fi

    ELAPSED=$(echo "$STATUS" | jq -r '.elapsed_seconds')
    PROGRESS=$(echo "$STATUS" | jq -r '.progress')

    if [ "$STATE" = "completed" ]; then
        # Clear the progress line and move to new line
        printf "\r\033[K"
        echo ""
        echo "✅ Job completed!"
        DOWNLOAD_URL=$(echo "$STATUS" | jq -r '.download_url')
        OUTPUT_SIZE=$(echo "$STATUS" | jq -r '.output_size_mb')
        echo "📊 Output: ${OUTPUT_SIZE} MB"
        echo ""

        if [ -z "$DOWNLOAD_URL" ] || [ "$DOWNLOAD_URL" = "null" ]; then
            echo "❌ Backend did not return a download URL"
            echo "$STATUS" | jq 2>/dev/null || echo "$STATUS"
            exit 1
        fi

        echo "📥 Downloading..."
        echo "🔗 $DOWNLOAD_URL"
        curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 \
          "$DOWNLOAD_URL" -o "$OUTPUT_FILE"

        if [ -f "$OUTPUT_FILE" ]; then
            FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
            echo "✅ Saved to: $OUTPUT_FILE ($FILE_SIZE)"
            echo "⏱️  Total time: ${ELAPSED} seconds"
            echo "Upscaled_url: $DOWNLOAD_URL"

            # Clickable hyperlink for modern terminals (VS Code, iTerm2, etc.)
            printf "\033]8;;%s\033\\Open in browser\033]8;;\033\\\n" "$DOWNLOAD_URL"

            # Auto-open browser if requested
            if [ "$AUTO_OPEN" = true ] || [ "$UPSCALE_OPEN" = "1" ]; then
                if command -v xdg-open >/dev/null 2>&1; then
                    xdg-open "$DOWNLOAD_URL" >/dev/null 2>&1 &
                elif command -v open >/dev/null 2>&1; then
                    open "$DOWNLOAD_URL" >/dev/null 2>&1 &
                elif command -v wslview >/dev/null 2>&1; then
                    wslview "$DOWNLOAD_URL" >/dev/null 2>&1 &
                elif command -v start >/dev/null 2>&1; then
                    start "" "$DOWNLOAD_URL" >/dev/null 2>&1 &
                else
                    echo "ℹ️  Could not auto-open (no xdg-open/open/wslview/start). Copy the URL above."
                fi
            fi
        else
            echo "❌ Download failed"
            exit 1
        fi
        break

    elif [ "$STATE" = "failed" ]; then
        # Clear the progress line
        printf "\r\033[K"
        ERROR=$(echo "$STATUS" | jq -r '.error')
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
