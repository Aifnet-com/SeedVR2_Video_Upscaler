#!/bin/bash

# SeedVR2 Video Upscaler - Simple CLI wrapper
# Usage: ./upscale.sh "https://example.com/video.mp4" [--resolution 720p|1080p|2k|4k]
#
# New behavior:
# - If the job appears "stuck waiting to be registered", we run a CDN watchdog:
#   check the expected Bunny CDN path 10x every 30s. If the file shows up, we
#   declare success; otherwise we fail the job.
#
# You can override the CDN path with:
#   export BUNNY_BASE_URL="https://aifnet.b-cdn.net"
#   export BUNNY_ROOT_DIR="tests/seedvr2_results"
#
# Defaults (sane for Aifnet deployment) are provided below.

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

# Optional overrides for CDN probe
BUNNY_BASE_URL="${BUNNY_BASE_URL:-https://aifnet.b-cdn.net}"
BUNNY_ROOT_DIR="${BUNNY_ROOT_DIR:-tests/seedvr2_results}"

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

# Helper: derive expected CDN URL for the output artifact
#   Take basename of VIDEO_URL path, append _<resolution>.mp4,
#   and place under $BUNNY_BASE_URL/$BUNNY_ROOT_DIR/
derive_cdn_url() {
    local input_url="$1"
    local res="$2"
    local base_name
    base_name="$(basename "$input_url")"
    if [ -z "$base_name" ] || [ "$base_name" = "/" ]; then
        base_name="video.mp4"
    fi
    local stem="${base_name%.*}"
    local file="${stem}_${res}.mp4"
    echo "${BUNNY_BASE_URL%/}/${BUNNY_ROOT_DIR:+/$BUNNY_ROOT_DIR}/$file" | sed 's#//#/#g' | sed 's#https:/#https://#'
}

# CDN watchdog: HEAD the CDN URL and return http code
cdn_http_code() {
    local url="$1"
    curl -sI -o /dev/null -w "%{http_code}" "$url"
}

# If we get "not yet registered", run a 10x30s CDN check loop
registration_watchdog() {
    local cdn_url="$1"
    echo "⏳ Waiting for job to be registered..."
    echo "🔎 CDN watchdog: probing $cdn_url"
    local tries=10
    local interval=30
    local i=1
    while [ $i -le $tries ]; do
        local code
        code=$(cdn_http_code "$cdn_url")
        if [ "$code" = "200" ]; then
            echo ""
            echo "✅ File appeared on CDN via watchdog!"
            echo "Upscaled_url: $cdn_url"
            echo "ℹ️  Note: job registration never materialized, but the artifact exists."
            return 0
        fi
        printf "\r\033[K⌛ Probe %d/%d: not found yet (HTTP %s). Next in %ss..." "$i" "$tries" "${code:-0}" "$interval"
        sleep $interval
        i=$((i+1))
    done
    echo ""
    echo "❌ Job never registered and no file appeared on CDN after $tries probes."
    return 1
}

# Poll status
while true; do
    STATUS=$(curl -s "$API_URL/status/$JOB_ID")
    STATE=$(echo $STATUS | jq -r '.status' 2>/dev/null)

    # Handle initial sync delay (404s / null state)
    if [ -z "$STATE" ] || [ "$STATE" = "null" ]; then
        # Derive expected CDN path and run watchdog immediately
        CDN_URL="$(derive_cdn_url "$VIDEO_URL" "$RESOLUTION")"
        if registration_watchdog "$CDN_URL"; then
            # Consider this a success path; print minimal stats and exit
            echo "✅ Job completed (detected via CDN watchdog)."
            exit 0
        else
            exit 1
        fi
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
        printf "\r\033[K⏳ Status: $STATE [$PROGRESS] - Elapsed: ${MINS}m ${SECS}s"
    fi

    sleep 5
done
