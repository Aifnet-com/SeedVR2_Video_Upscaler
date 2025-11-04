#!/bin/bash

# SeedVR2 Video Upscaler - CLI wrapper (BunnyCDN FTP Storage)
# Usage: ./upscale.sh "https://example.com/video.mp4" [--resolution 720p|1080p|2k|4k]

if [ -z "$1" ]; then
    echo "Usage: $0 <video_url> [--resolution 720p|1080p|2k|4k]"
    echo "Example: $0 'https://example.com/video.mp4'"
    echo "Example: $0 'https://example.com/video.mp4' --resolution 720p"
    echo "Example: $0 'https://example.com/video.mp4' --resolution 4k"
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

# Submit job
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

    # Handle initial sync delay
    if [ -z "$STATE" ] || [ "$STATE" = "null" ]; then
        printf "\r\033[K⏳ Waiting for job to be registered..."
        sleep 2
        continue
    fi

    ELAPSED=$(echo $STATUS | jq -r '.elapsed_seconds')
    PROGRESS=$(echo $STATUS | jq -r '.progress')

    if [ "$STATE" = "completed" ]; then
        # Clear the progress line
        printf "\r\033[K"
        echo ""
        echo "✅ Job completed!"

        # Get CDN URL and filename
        CDN_URL=$(echo $STATUS | jq -r '.cdn_url')
        OUTPUT_SIZE=$(echo $STATUS | jq -r '.output_size_mb')
        FILENAME=$(echo $STATUS | jq -r '.filename')

        echo "📊 Output: ${OUTPUT_SIZE} MB"
        echo "📁 Filename: ${FILENAME}"
        echo "🔗 CDN URL: ${CDN_URL}"
        echo ""

        echo "📥 Downloading from CDN..."
        curl -s "$CDN_URL" -o "$OUTPUT_FILE"

        if [ -f "$OUTPUT_FILE" ]; then
            FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

            # Check if file is actually a video (>100KB)
            BYTE_SIZE=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE" 2>/dev/null)

            if [ "$BYTE_SIZE" -gt 100000 ]; then
                echo "✅ Saved to: $OUTPUT_FILE ($FILE_SIZE)"
                echo "⏱️  Total time: ${ELAPSED} seconds"
                echo ""
                echo "💡 Direct link: ${CDN_URL}"
            else
                echo "❌ Download failed (file too small: $FILE_SIZE)"
                echo "💡 But the video should be available at: ${CDN_URL}"
                exit 1
            fi
        else
            echo "❌ Download failed"
            echo "💡 But the video should be available at: ${CDN_URL}"
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

        # Clear line, then print status
        printf "\r\033[K⏳ Status: $STATE [$PROGRESS] - Elapsed: ${MINS}m ${SECS}s"
    fi

    sleep 5
done