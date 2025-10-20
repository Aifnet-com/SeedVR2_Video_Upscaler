#!/bin/bash

# SeedVR2 Video Upscaler - Complete workflow in one command
# Usage: ./upscale.sh "https://example.com/video.mp4" [output.mp4]

if [ -z "$1" ]; then
    echo "Usage: $0 <video_url> [output_filename]"
    echo "Example: $0 'https://astra.app/api/files/xxx.mp4' output.mp4"
    exit 1
fi

VIDEO_URL="$1"
OUTPUT_FILE="${2:-upscaled_result.mp4}"
API_URL="https://aifnet--seedvr2-upscaler-fastapi-app.modal.run"

echo "🚀 Submitting upscaling job..."

# Submit job
RESPONSE=$(curl -s -X POST "$API_URL/upscale" \
  -H "Content-Type: application/json" \
  -d "{\"video_url\": \"$VIDEO_URL\"}")

JOB_ID=$(echo $RESPONSE | jq -r '.job_id')

if [ -z "$JOB_ID" ] || [ "$JOB_ID" = "null" ]; then
    echo "❌ Failed to submit job"
    echo $RESPONSE | jq
    exit 1
fi

echo "✅ Job submitted: $JOB_ID"
echo ""

# Poll status
while true; do
    STATUS=$(curl -s "$API_URL/status/$JOB_ID")
    STATE=$(echo $STATUS | jq -r '.status')
    ELAPSED=$(echo $STATUS | jq -r '.elapsed_seconds')
    PROGRESS=$(echo $STATUS | jq -r '.progress')
    
    if [ "$STATE" = "completed" ]; then
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
        ERROR=$(echo $STATUS | jq -r '.error')
        echo "❌ Job failed: $ERROR"
        exit 1
        
    else
        MINS=$((${ELAPSED%.*} / 60))
        SECS=$((${ELAPSED%.*} % 60))
        printf "\r⏳ Status: $STATE [$PROGRESS] - Elapsed: ${MINS}m ${SECS}s"
    fi
    
    sleep 5
done
