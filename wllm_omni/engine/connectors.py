from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from wllm_omni.model_types import ModelParadigm
from wllm_omni.engine.planning import DiffusionPlan, merge_negative_prompt
from wllm_omni.models.ar_pipeline import ARTextOutput
from wllm_omni.request import OmniRequest
from wllm_omni.sampling_params import clone_sampling_params

from wllm_omni.engine.stage import StageOutput


@dataclass(slots=True)
class ConnectorContext:
    root_request: OmniRequest
    source_node: str
    target_node: str
    source_output: StageOutput


class StageConnector(ABC):
    """Transforms an upstream stage output into a downstream stage request."""

    @abstractmethod
    def connect(self, context: ConnectorContext) -> OmniRequest:
        pass


class ARToDiffusionConnector(StageConnector):
    """Bridge AR text output into a Wan image-to-video diffusion request."""

    def connect(self, context: ConnectorContext) -> OmniRequest:
        if not isinstance(context.source_output.data, ARTextOutput):
            raise TypeError(
                "ARToDiffusionConnector expects ARTextOutput, "
                f"got {type(context.source_output.data).__name__}."
            )
        ar_text = context.source_output.data.text
        ar_metadata = _merged_ar_metadata(context.source_output)
        plan = DiffusionPlan.from_text(ar_text)
        sampling = clone_sampling_params(context.root_request.sampling_params)
        fallback_reason = None
        ar_guidance = ''
        if not plan.parsed and _looks_like_broken_structured_output(ar_text):
            diffusion_prompt = context.root_request.prompt
            fallback_reason = 'invalid_structured_ar_output'
        elif _looks_like_truncated_ar_guidance(ar_text, ar_metadata):
            diffusion_prompt = context.root_request.prompt
            fallback_reason = 'truncated_ar_guidance'
        else:
            ar_guidance = plan.render_prompt()
            diffusion_prompt = _merge_root_prompt_with_ar_guidance(
                context.root_request.prompt,
                ar_guidance,
            )
            sampling.negative_prompt = merge_negative_prompt(
                sampling.negative_prompt,
                plan.render_negative_prompt_additions(),
            )
        return OmniRequest(
            prompt=diffusion_prompt,
            image=context.root_request.image,
            sampling_params=sampling,
            model_paradigm=ModelParadigm.DIFFUSION,
            request_id=context.root_request.request_id,
            extra={
                'bridge': 'ar_text_to_diffusion_prompt',
                'ar_output_text': ar_text,
                'ar_guidance_prompt': ar_guidance,
                'diffusion_prompt': diffusion_prompt,
                'diffusion_plan': plan.to_metadata(),
                'bridge_strategy': 'preserve_root_prompt_with_ar_guidance',
                'bridge_parse_success': plan.parsed,
                'bridge_fallback': fallback_reason is not None,
                'bridge_fallback_reason': fallback_reason,
            },
        )


def _merged_ar_metadata(output: StageOutput) -> dict[str, object]:
    metadata = dict(getattr(output.data, 'metadata', {}) or {})
    metadata.update(output.metadata)
    return metadata


def _looks_like_broken_structured_output(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    return (
        stripped.startswith('```')
        or stripped.startswith('{')
        or '"prompt"' in lowered
        or 'negative_prompt_additions' in lowered
        or lowered.startswith('json')
    )


def _looks_like_truncated_ar_guidance(text: str, metadata: dict[str, object]) -> bool:
    if metadata.get('stop_reason') == 'token_budget':
        return True
    stripped = text.strip().rstrip('.,;:')
    if not stripped:
        return True
    tail = stripped.split()[-1].lower()
    return tail in {'and', 'or', 'with', 'without', 'of', 'for', 'to', 'in', 'on', 'at', 'by', 'from'}


def _merge_root_prompt_with_ar_guidance(root_prompt: str, ar_guidance: str) -> str:
    root = ' '.join(root_prompt.strip().split())
    guidance = ' '.join(ar_guidance.strip().split())
    if not root:
        return guidance
    if not guidance:
        return root
    if guidance.lower() in root.lower():
        return root
    return f'{root} Additional motion/camera guidance from AR: {guidance}.'


class CallableARToDiffusionConnector(StageConnector):
    """Compatibility wrapper for older AR-output connector callables."""

    def __init__(self, connector: Callable[[OmniRequest, ARTextOutput], OmniRequest]):
        self.connector = connector

    def connect(self, context: ConnectorContext) -> OmniRequest:
        if not isinstance(context.source_output.data, ARTextOutput):
            raise TypeError(
                "CallableARToDiffusionConnector expects ARTextOutput, "
                f"got {type(context.source_output.data).__name__}."
            )
        return self.connector(context.root_request, context.source_output.data)
