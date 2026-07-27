from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from PIL import Image

from wllm_omni.config import DEFAULT_IMAGE, DEFAULT_PROMPT
from wllm_omni.model_types import ModelParadigm
from wllm_omni.sampling_params import OmniSamplingParams, PRESETS, clone_sampling_params


@dataclass(slots=True)
class OmniRequest:
    prompt: str = DEFAULT_PROMPT
    image: str | Path | Image.Image = DEFAULT_IMAGE
    sampling_params: OmniSamplingParams = field(default_factory=lambda: clone_sampling_params(PRESETS["quality"]))
    model_paradigm: ModelParadigm = ModelParadigm.DIFFUSION
    request_id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def batch_key(self) -> tuple:
        sp = self.sampling_params
        return (
            sp.height,
            sp.width,
            sp.num_frames,
            sp.num_inference_steps,
            sp.guidance_scale,
            sp.flow_shift,
            sp.negative_prompt,
            sp.fps,
        )

    @property
    def prompt_cache_key(self) -> tuple:
        sp = self.sampling_params
        return (self.prompt, sp.negative_prompt)
