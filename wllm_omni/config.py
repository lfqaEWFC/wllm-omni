from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight import path
    class _TorchStub:
        dtype = object
        bfloat16 = "bfloat16"
        float32 = "float32"

    torch = _TorchStub()


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = str(REPO_ROOT / "models" / "Wan2.2-TI2V-5B-Diffusers")
DEFAULT_IMAGE = str(REPO_ROOT / "assets" / "image.png")
DEFAULT_PROMPT = (
    "Warm outdoor pet portrait, a young golden retriever puppy sitting on a vibrant green lawn in a park. "
    "The puppy faces the camera with bright dark eyes, relaxed floppy ears, an open smiling mouth, and a "
    "pink tongue. Soft natural daylight illuminates its fluffy cream-golden fur, with a shallow depth of "
    "field blurring trees and grass in the background. Over a longer natural shot, the puppy blinks, "
    "gently tilts its head, pants happily with its tongue out, shifts its front paw forward, leans closer "
    "with curious eye contact, and lets its fluffy ears and chest fur move slightly in a light breeze. "
    "Low eye-level close-up composition, stable camera, friendly and lively atmosphere, high quality "
    "realistic video."
)
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards, ghosting, double image, "
    "duplicate subject, motion trails, afterimage, frame blending"
)

PIPELINE_WAN_I2V = "wan_i2v"
PIPELINE_AR_TEXT = "ar_text"
PIPELINE_QWEN_TO_WAN_I2V = "qwen_to_wan_i2v"
SUPPORTED_PIPELINES = (PIPELINE_WAN_I2V, PIPELINE_AR_TEXT, PIPELINE_QWEN_TO_WAN_I2V)

AR_TEXT_NODE = "ar.text_generation"
AR_PROMPT_BRIDGE_NODE = "ar.prompt_bridge"
DIFFUSION_WAN_I2V_NODE = "diffusion.wan22_i2v"

AR_PROMPT_MODE_TEXT = "text"
AR_PROMPT_MODE_I2V_BRIDGE = "i2v_bridge"
SUPPORTED_AR_PROMPT_MODES = (AR_PROMPT_MODE_TEXT, AR_PROMPT_MODE_I2V_BRIDGE)


@dataclass(slots=True)
class EngineConfig:
    model: str = DEFAULT_MODEL
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    # Wan VAE stays in fp32 by default for decode stability.
    vae_dtype: torch.dtype = torch.float32
    local_files_only: bool = True
    use_cpu_offload: bool = True
    max_num_seqs: int = 2
    prompt_cache_size: int = 8
    image_cache_size: int = 4
    condition_cache_size: int = 4
    enable_profiling: bool = False
    probe_condition_cache: bool = False
    enable_mini_omni: bool = False
    pipeline: str = PIPELINE_WAN_I2V
    ar_model: str | None = None
