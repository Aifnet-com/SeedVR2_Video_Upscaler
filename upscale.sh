#!/bin/bash

# SeedVR2 Video Upscaler - Complete workflow in one command
# Supports both URLs and local files
# Usage: ./upscale.sh "https://example.com/video.mp4" [output.mp4]
#        ./upscale.sh "/path/to/local/video.mp4" [output.mp4]

if [ -z "$1" ]; then
    echo "Usage: $0 <video_url_or_path> [output_filename]"
    echo ""
    echo "Examples:"
    echo "  $0 'https://astra.app/api/files/xxx.mp4' output.mp4"
    echo "  $0 '/home/user/video.mp4' output.mp4"
    exit 1
fi

VIDEO_SOURCE="$1"
API_URL="https://aifnet--seedvr2-upscaler-fastapi-app.modal.run"

# Generate unique output filename if not provided
if [ -z "$2" ]; then
    OUTPUT_FILE="upscaled_temp.mp4"
else
    OUTPUT_FILE="$2"
fi

echo "🚀 Submitting upscaling job..."

# Determine if input is URL or local file
if [[ "$VIDEO_SOURCE" == http* ]]; then
    # It's a URL
    echo "📥 Input: URL"
    REQUEST_BODY="{\"video_url\": \"$VIDEO_SOURCE\"}"
else
    # It's a local file
    if [ ! -f "$VIDEO_SOURCE" ]; then
        echo "❌ File not found: $VIDEO_SOURCE"
        exit 1
    fi
    
    echo "📥 Input: Local file ($VIDEO_SOURCE)"
    echo "🔄 Encoding to base64..."
    VIDEO_B64=$(base64 -w 0 "$VIDEO_SOURCE")
    REQUEST_BODY="{\"video_base64\": \"$VIDEO_B64\"}"
    echo "✅ Encoded"
fi

# Submit job
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

# Use first 8 chars of job ID for unique filename if no custom name provided
if [ "$OUTPUT_FILE" = "upscaled_temp.mp4" ]; then
    SHORT_ID=${JOB_ID:0:8}
    OUTPUT_FILE="upscaled_${SHORT_ID}.mp4"
fi

echo "📁 Output will be saved to: $OUTPUT_FILE"
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
