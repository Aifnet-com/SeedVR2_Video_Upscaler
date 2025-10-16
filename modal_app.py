"""
Modal Serverless deployment for SeedVR2 Video Upscaler
"""

import modal

# Create Modal app
app = modal.App("seedvr2-upscaler")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg", "git")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0", 
        "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "opencv-python-headless==4.10.0.84",
        "numpy>=1.26.4",
        "safetensors>=0.4.5",
        "einops",
        "omegaconf>=2.3.0",
        "diffusers>=0.34.0",
        "pytorch-extension",
        "rotary_embedding_torch",
        "peft>=0.15.0",
        "transformers>=4.46.3",
        "accelerate>=1.1.1",
        "huggingface-hub>=0.26.2",
        "requests>=2.32.3"
    )
)

# Create persistent volume for model weights
volume = modal.Volume.from_name("seedvr2-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",  # or "A100-80GB", "H100" for more VRAM
    timeout=1000,  # 1 hour max
    volumes={"/models": volume},
)
def upscale_video(
    video_url: str,
    batch_size: int = 100,
    temporal_overlap: int = 12,
    stitch_mode: str = "crossfade",
    model: str = "seedvr2_ema_7b_fp16.safetensors"
):
    """
    Upscale a video using SeedVR2
    
    Args:
        video_url: URL of the video to upscale
        batch_size: Number of frames per batch (default: 100)
        temporal_overlap: Frames to overlap between batches (default: 12)
        stitch_mode: Stitching mode - 'later' or 'crossfade' (default: crossfade)
        model: Model to use (default: seedvr2_ema_7b_fp16.safetensors)
    
    Returns:
        Dict with video bytes, logs, and sizes
    """
    import subprocess
    import tempfile
    import os
    import requests
    import sys
    
    # Add workspace to path for imports
    sys.path.insert(0, "/root")
    
    print(f"🚀 Starting SeedVR2 upscaling")
    print(f"📋 Config: batch_size={batch_size}, overlap={temporal_overlap}, mode={stitch_mode}")
    
    # Set CUDA environment
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    
    # Copy inference_cli.py content inline since we can't mount files
    # We'll need to fetch it from GitHub
    inference_script = """
# This will be replaced with actual inference_cli.py content
# For now, we'll download it at runtime
"""
    
    # Download inference_cli.py from your GitHub repo
    cli_url = "https://raw.githubusercontent.com/gkirilov7/ComfyUI-SeedVR2_VideoUpscaler/main/inference_cli.py"
    response = requests.get(cli_url)
    with open("/root/inference_cli.py", "w") as f:
        f.write(response.text)
    
    # Also download src directory files (we need the whole repo)
    # Clone the repo instead
    subprocess.run(["git", "clone", "https://github.com/gkirilov7/ComfyUI-SeedVR2_VideoUpscaler.git", "/root/repo"], check=True)
    os.chdir("/root/repo")
    
    # Download input video
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mp4")
        output_path = os.path.join(tmpdir, "output.mp4")
        
        print(f"📥 Downloading video from {video_url}")
        response = requests.get(video_url, stream=True, timeout=300)
        response.raise_for_status()
        
        with open(input_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        print(f"📦 Input video size: {input_size_mb:.2f} MB")
        
        # Run inference
        cmd = [
            "python", "inference_cli.py",
            "--video_path", input_path,
            "--batch_size", str(batch_size),
            "--temporal_overlap", str(temporal_overlap),
            "--stitch_mode", stitch_mode,
            "--model", model,
            "--model_dir", "/models",
            "--output", output_path,
            "--debug"
        ]
        
        print(f"🔧 Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/root/repo")
        
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            raise Exception(f"Upscaling failed: {result.stderr}")
        
        print(result.stdout)
        
        if not os.path.exists(output_path):
            raise Exception("Output file not created")
        
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Output video size: {output_size_mb:.2f} MB")
        
        # Read output file
        with open(output_path, 'rb') as f:
            output_data = f.read()
        
        return {
            "video": output_data,
            "logs": result.stdout,
            "input_size_mb": input_size_mb,
            "output_size_mb": output_size_mb
        }


@app.local_entrypoint()
def main(
    video_url: str,
    batch_size: int = 100,
    temporal_overlap: int = 12,
    stitch_mode: str = "crossfade",
    model: str = "seedvr2_ema_7b_fp16.safetensors",
    output_file: str = "output_upscaled.mp4"
):
    """
    CLI entry point for testing
    
    Usage:
        modal run modal_app.py --video-url "https://example.com/video.mp4"
    """
    result = upscale_video.remote(
        video_url=video_url,
        batch_size=batch_size,
        temporal_overlap=temporal_overlap,
        stitch_mode=stitch_mode,
        model=model
    )
    
    # Save output video
    with open(output_file, 'wb') as f:
        f.write(result["video"])
    
    print(f"\n✅ Upscaling complete!")
    print(f"📁 Output saved to: {output_file}")
    print(f"📊 Input: {result['input_size_mb']:.2f} MB → Output: {result['output_size_mb']:.2f} MB")
