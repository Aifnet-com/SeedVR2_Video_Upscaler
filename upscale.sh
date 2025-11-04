#!/bin/bash

# SeedVR2 Video Upscaler - Simple CLI wrapper
# Usage: ./upscale.sh "https://example.com/video.mp4" [--resolution 720p|1080p|2k|4k] [--open|--no-open]
#
# Notes:
# - Prints a clickable URL (OSC-8, two variants) + the raw URL.
# - Auto-opens the URL if a desktop session is detected (can disable via --no-open or UPSCALE_OPEN=0).
# - Force opening even over SSH by passing --open or setting UPSCALE_OPEN=1.

set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <video_url> [--resolution 720p|1080p|2k|4k] [--open|--no-open]"
  echo "Example: $0 'https://astra.app/api/files/xxx.mp4'"
  echo "Example: $0 'https://astra.app/api/files/xxx.mp4' --resolution 720p"
  echo "Example: $0 'https://astra.app/api/files/xxx.mp4' --resolution 4k --open"
  exit 1
fi

VIDEO_URL="$1"; shift || true
API_URL="https://aifnet--seedvr2-upscaler-fastapi-app.modal.run"
RESOLUTION="1080p"

# AUTO_OPEN modes:
#   "auto"  -> open if desktop present and UPSCALE_OPEN != 0 (default)
#   "force" -> open regardless of desktop (if an opener exists)
#   "off"   -> never open
AUTO_OPEN="auto"

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resolution)
      RESOLUTION="${2:-}"; shift 2 || true
      ;;
    --open)
      AUTO_OPEN="force"; shift
      ;;
    --no-open)
      AUTO_OPEN="off"; shift
      ;;
    *)
      # ignore unknown args for forward compatibility
      shift
      ;;
  esac
done

if [[ ! "$RESOLUTION" =~ ^(720p|1080p|2k|4k)$ ]]; then
  echo "❌ Invalid resolution: $RESOLUTION. Must be 720p, 1080p, 2k, or 4k"
  exit 1
fi

echo "🚀 Submitting upscaling job..."

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

JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id' 2>/dev/null || true)

if [ -z "$JOB_ID" ] || [ "$JOB_ID" = "null" ]; then
  echo "❌ Failed to submit job"
  echo "$RESPONSE" | jq 2>/dev/null || echo "$RESPONSE"
  exit 1
fi

echo "✅ Job submitted: $JOB_ID"

GPU_TYPE=$(echo "$RESPONSE" | jq -r '.gpu_type' 2>/dev/null || true)
if [ -n "$GPU_TYPE" ] && [ "$GPU_TYPE" != "null" ]; then
  echo "🖥️  GPU: $GPU_TYPE"
fi

SHORT_ID=${JOB_ID:0:8}
OUTPUT_FILE="upscaled_${SHORT_ID}.mp4"
echo "📁 Output will be saved to: $OUTPUT_FILE"
echo ""

# Poll status
while true; do
  STATUS=$(curl -s "$API_URL/status/$JOB_ID")
  STATE=$(echo "$STATUS" | jq -r '.status' 2>/dev/null || true)

  if [ -z "$STATE" ] || [ "$STATE" = "null" ]; then
    printf "\r\033[K⏳ Waiting for job to be registered..."
    sleep 2
    continue
  fi

  ELAPSED=$(echo "$STATUS" | jq -r '.elapsed_seconds' 2>/dev/null || echo "0")
  PROGRESS=$(echo "$STATUS" | jq -r '.progress' 2>/dev/null || echo "")

  if [ "$STATE" = "completed" ]; then
    printf "\r\033[K"
    echo ""
    echo "✅ Job completed!"
    DOWNLOAD_URL=$(echo "$STATUS" | jq -r '.download_url' 2>/dev/null || true)
    OUTPUT_SIZE=$(echo "$STATUS" | jq -r '.output_size_mb' 2>/dev/null || echo "")
    echo "📊 Output: ${OUTPUT_SIZE} MB"
    echo ""

    if [ -z "$DOWNLOAD_URL" ] || [ "$DOWNLOAD_URL" = "null" ]; then
      echo "❌ Backend did not return a download URL"
      echo "$STATUS" | jq 2>/dev/null || echo "$STATUS"
      exit 1
    fi

    echo "📥 Downloading..."
    echo "🔗 $DOWNLOAD_URL"
    # Robust download: follow redirects, fail on HTTP errors, retry briefly
    curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 \
      "$DOWNLOAD_URL" -o "$OUTPUT_FILE"

    if [ -f "$OUTPUT_FILE" ]; then
      FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
      echo "✅ Saved to: $OUTPUT_FILE ($FILE_SIZE)"
      echo "⏱️  Total time: ${ELAPSED} seconds"
      echo "Upscaled_url: $DOWNLOAD_URL"

      # Clickable hyperlinks (emit both ST and BEL variants for wider terminal support)
      printf "\033]8;;%s\033\\Open in browser\033]8;;\033\\\n" "$DOWNLOAD_URL"
      printf "\033]8;;%s\aOpen in browser (alt)\033]8;;\a\n" "$DOWNLOAD_URL"
      # Always print the plain URL too
      echo "$DOWNLOAD_URL"

      # Decide whether to auto-open
      SHOULD_OPEN=false
      if [ "$AUTO_OPEN" = "force" ]; then
        SHOULD_OPEN=true
      elif [ "$AUTO_OPEN" = "auto" ]; then
        # Open if on a desktop AND UPSCALE_OPEN is not "0" (default allow)
        if { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ] || [ "$(uname -s)" = "Darwin" ]; } && [ "${UPSCALE_OPEN:-1}" != "0" ]; then
          SHOULD_OPEN=true
        fi
      fi

      if $SHOULD_OPEN; then
        if command -v xdg-open >/dev/null 2>&1; then
          xdg-open "$DOWNLOAD_URL" >/dev/null 2>&1 &
        elif command -v open >/dev/null 2>&1; then
          open "$DOWNLOAD_URL" >/dev/null 2>&1 &
        elif command -v wslview >/dev/null 2>&1; then
          wslview "$DOWNLOAD_URL" >/dev/null 2>&1 &
        elif command -v start >/dev/null 2>&1; then
          start "" "$DOWNLOAD_URL" >/dev/null 2>&1 &
        else
          echo "ℹ️  Could not auto-open (no xdg-open/open/wslview/start)."
        fi
      fi
    else
      echo "❌ Download failed"
      exit 1
    fi
    break

  elif [ "$STATE" = "failed" ]; then
    printf "\r\033[K"
    ERROR=$(echo "$STATUS" | jq -r '.error' 2>/dev/null || echo "Unknown error")
    echo "❌ Job failed: $ERROR"
    exit 1

  else
    MINS=$((${ELAPSED%.*} / 60))
    SECS=$((${ELAPSED%.*} % 60))
    printf "\r\033[K⏳ Status: %s [%s] - Elapsed: %sm %ss" "$STATE" "$PROGRESS" "$MINS" "$SECS"
  fi

  sleep 5
done
