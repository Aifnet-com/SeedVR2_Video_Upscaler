import fal
from pydantic import BaseModel, Field
from fal.toolkit import File
import subprocess
import tempfile
import os
from pathlib import Path

class Input(BaseModel):
    video_url: str = Field(description="URL of the video to upscale")
    batch_size: int = Field(default=100, description="Number of frames per batch")
    temporal_overlap: int = Field(default=12, description="Frames to overlap between batches")
    stitch_mode: str = Field(default="crossfade", description="Stitching mode: later or crossfade")
    model: str = Field(default="seedvr2_ema_7b_fp16.safetensors", description="Model to use")

class Output(BaseModel):
    video: File = Field(description="Upscaled video file")
    logs: str = Field(description="Processing logs")

class SeedVR2Upscaler(fal.App):
    machine_type = "GPU-A100-40GB"
    keep_alive = 300
    requirements = [
        "torch==2.6.0",
        "torchvision==2.21.0",
        "torchaudio==2.6.0",
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
    ]
    
    def setup(self):
        """Setup model cache directory"""
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
        self.model_cache_dir = "/data/models"
        os.makedirs(self.model_cache_dir, exist_ok=True)
        print(f"✅ Model cache directory ready: {self.model_cache_dir}")
    
    @fal.endpoint("/")
    def upscale(self, input: Input) -> Output:
        """Main upscaling endpoint"""
        print(f"🚀 Starting upscaling: {input.video_url}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")
            
            # Download input video
            print("📥 Downloading video...")
            import requests
            response = requests.get(input.video_url, stream=True)
            response.raise_for_status()
            with open(input_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Run inference
            cmd = [
                "python", "inference_cli.py",
                "--video_path", input_path,
                "--batch_size", str(input.batch_size),
                "--temporal_overlap", str(input.temporal_overlap),
                "--stitch_mode", input.stitch_mode,
                "--model", input.model,
                "--model_dir", self.model_cache_dir,
                "--output", output_path,
                "--debug"
            ]
            
            print(f"⚙️ Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Upscaling failed: {result.stderr}")
            
            print("✅ Upscaling complete!")
            
            return Output(
                video=File.from_path(output_path),
                logs=result.stdout
            )
