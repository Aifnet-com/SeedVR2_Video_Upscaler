import os
import tempfile
import subprocess
from pathlib import Path
import requests

def download_video(url: str, output_path: str):
    """Download video from URL"""
    print(f"📥 Downloading video from {url}")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                progress = (downloaded / total_size) * 100
                print(f"📊 Download progress: {progress:.1f}%")
    
    print(f"✅ Download complete: {output_path}")

def handler(context):
    """
    Main fal.ai handler function
    
    Expected input JSON:
    {
        "video_url": "https://example.com/input.mp4",
        "batch_size": 100,
        "temporal_overlap": 12,
        "stitch_mode": "crossfade",
        "model": "seedvr2_ema_7b_fp16.safetensors"
    }
    """
    # Parse input
    video_url = context["video_url"]
    batch_size = context.get("batch_size", 100)
    temporal_overlap = context.get("temporal_overlap", 12)
    stitch_mode = context.get("stitch_mode", "crossfade")
    model = context.get("model", "seedvr2_ema_7b_fp16.safetensors")
    
    print(f"🚀 Starting SeedVR2 upscaling")
    print(f"📋 Config: batch_size={batch_size}, overlap={temporal_overlap}, mode={stitch_mode}, model={model}")
    
    # Setup model cache directory (persistent storage)
    model_cache_dir = os.environ.get("MODEL_CACHE_DIR", "/data/models")
    os.makedirs(model_cache_dir, exist_ok=True)
    
    # Create temp directories
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mp4")
        output_path = os.path.join(tmpdir, "output.mp4")
        
        # Download input video
        download_video(video_url, input_path)
        
        # Check input file size
        input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        print(f"📦 Input video size: {input_size_mb:.2f} MB")
        
        # Run inference
        print(f"🎬 Starting upscaling...")
        cmd = [
            "python", "inference_cli.py",
            "--video_path", input_path,
            "--batch_size", str(batch_size),
            "--temporal_overlap", str(temporal_overlap),
            "--stitch_mode", stitch_mode,
            "--model", model,
            "--model_dir", model_cache_dir,
            "--output", output_path,
            "--debug"
        ]
        
        print(f"🔧 Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Upscaling failed!")
            print(f"STDERR: {result.stderr}")
            raise Exception(f"Upscaling failed: {result.stderr}")
        
        print(result.stdout)
        
        # Check output file
        if not os.path.exists(output_path):
            raise Exception(f"Output file not created: {output_path}")
        
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Upscaling complete! Output size: {output_size_mb:.2f} MB")
        
        # Return output (fal.ai automatically uploads files)
        return {
            "video": Path(output_path),
            "logs": result.stdout,
            "input_size_mb": input_size_mb,
            "output_size_mb": output_size_mb
        }
