"""Pipeline-local policies and AR-to-diffusion plan parsing.

V0 uses explicit pipeline names instead of prompt-driven graph synthesis.
The planner in this module maps a selected pipeline to a static stage
graph and stage-local policies; connector-side DiffusionPlan parsing only
normalizes optional AR guidance before it is merged into the diffusion
request.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

from wllm_omni.config import (
    AR_PROMPT_BRIDGE_NODE,
    AR_PROMPT_MODE_I2V_BRIDGE,
    AR_PROMPT_MODE_TEXT,
    AR_TEXT_NODE,
    DIFFUSION_WAN_I2V_NODE,
    PIPELINE_AR_TEXT,
    PIPELINE_QWEN_TO_WAN_I2V,
    PIPELINE_WAN_I2V,
)


@dataclass(frozen=True, slots=True)
class ARStagePolicy:
    """Internal policy for one configured AR stage."""
    role: str
    prompt_mode: str
    output_contract: str
    allow_stream: bool
    token_budget: int | None = None


@dataclass(frozen=True, slots=True)
class MiniOmniPipelinePlan:
    """Static top-level plan selected by a user-facing pipeline name."""
    name: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...] = ()
    ar_policy: ARStagePolicy | None = None
    allow_stream: bool = False

    @property
    def requires_ar(self) -> bool:
        return self.ar_policy is not None

    @property
    def requires_diffusion(self) -> bool:
        return DIFFUSION_WAN_I2V_NODE in self.nodes


AR_TEXT_STAGE_POLICY = ARStagePolicy(
    role='text_generation',
    prompt_mode=AR_PROMPT_MODE_TEXT,
    output_contract='final_text',
    allow_stream=True,
    token_budget=None,
)

QWEN_TO_WAN_STAGE_POLICY = ARStagePolicy(
    role='prompt_bridge',
    prompt_mode=AR_PROMPT_MODE_I2V_BRIDGE,
    output_contract='supplemental_guidance',
    allow_stream=False,
    token_budget=64,
)


DEFAULT_PIPELINE_PLANS = (
    MiniOmniPipelinePlan(
        name=PIPELINE_AR_TEXT,
        nodes=(AR_TEXT_NODE,),
        ar_policy=AR_TEXT_STAGE_POLICY,
        allow_stream=True,
    ),
    MiniOmniPipelinePlan(
        name=PIPELINE_WAN_I2V,
        nodes=(DIFFUSION_WAN_I2V_NODE,),
    ),
    MiniOmniPipelinePlan(
        name=PIPELINE_QWEN_TO_WAN_I2V,
        nodes=(AR_PROMPT_BRIDGE_NODE, DIFFUSION_WAN_I2V_NODE),
        edges=((AR_PROMPT_BRIDGE_NODE, DIFFUSION_WAN_I2V_NODE),),
        ar_policy=QWEN_TO_WAN_STAGE_POLICY,
    ),
)


class MiniOmniPlanner:
    """Static planner for the single-request mini-Omni runtime.

    This is deliberately not a free-form graph generator. V0 maps an explicit
    pipeline name to a pre-registered stage graph and its AR stage policy. That
    mirrors the practical boundary used by vLLM-Omni-style systems: user-facing
    requests select an existing workflow, while execution details stay inside
    each stage scheduler / runner / executor.
    """

    def __init__(self, plans: tuple[MiniOmniPipelinePlan, ...] = DEFAULT_PIPELINE_PLANS):
        self._plans: dict[str, MiniOmniPipelinePlan] = {}
        for plan in plans:
            if not plan.nodes:
                raise ValueError(f"Pipeline plan {plan.name!r} requires at least one stage node.")
            if plan.name in self._plans:
                raise ValueError(f"Duplicate pipeline plan name={plan.name!r}.")
            self._plans[plan.name] = plan

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._plans)

    @property
    def plans(self) -> tuple[MiniOmniPipelinePlan, ...]:
        return tuple(self._plans.values())

    def plan_pipeline(self, pipeline: str) -> MiniOmniPipelinePlan:
        try:
            return self._plans[pipeline]
        except KeyError as exc:
            known = ', '.join(self.names)
            raise ValueError(f"Unsupported pipeline={pipeline!r}; supported pipelines: {known}.") from exc

    def ar_stage_policy(self, pipeline: str) -> ARStagePolicy:
        plan = self.plan_pipeline(pipeline)
        if plan.ar_policy is None:
            raise ValueError(f"Pipeline {pipeline!r} does not contain an AR stage.")
        return plan.ar_policy

    def require_streamable(self, pipeline: str) -> MiniOmniPipelinePlan:
        plan = self.plan_pipeline(pipeline)
        if not plan.allow_stream:
            raise RuntimeError(f"Pipeline {pipeline!r} does not support streaming in V0.")
        if len(plan.nodes) != 1:
            raise RuntimeError("Streaming V0 supports only single-stage pipeline plans.")
        return plan


def pipeline_plan_for_pipeline(pipeline: str) -> MiniOmniPipelinePlan:
    return MiniOmniPlanner().plan_pipeline(pipeline)


@dataclass(frozen=True, slots=True)
class DiffusionPlan:
    """Normalized connector-side view of AR output for diffusion.

    Plain AR text is accepted as supplemental guidance. JSON-shaped AR output is
    parsed only when it is complete and valid, allowing the connector to fall
    back to the root prompt for malformed structured output.
    """
    prompt: str
    negative_prompt_additions: tuple[str, ...] = ()
    style: str | None = None
    camera_motion: str | None = None
    motion_level: str | None = None
    profile: str | None = None
    subject: str | None = None
    action: str | None = None
    scene: str | None = None
    composition: str | None = None
    lighting: str | None = None
    raw_text: str = ''
    parsed: bool = False

    @classmethod
    def from_text(cls, text: str) -> 'DiffusionPlan':
        payload = _load_json_payload(text)
        if payload is None:
            normalized = _normalize_text(text)
            return cls(prompt=normalized or text.strip(), raw_text=text, parsed=False)

        prompt = _normalize_text(_get_string(payload, 'prompt'))
        if not prompt:
            prompt = _normalize_text(text)
        return cls(
            prompt=prompt,
            negative_prompt_additions=_normalize_string_list(payload.get('negative_prompt_additions')),
            style=_normalize_optional_text(payload.get('style')),
            camera_motion=_normalize_optional_text(payload.get('camera_motion')),
            motion_level=_normalize_optional_text(payload.get('motion_level')),
            profile=_normalize_optional_text(payload.get('profile')),
            subject=_normalize_optional_text(payload.get('subject')),
            action=_normalize_optional_text(payload.get('action')),
            scene=_normalize_optional_text(payload.get('scene')),
            composition=_normalize_optional_text(payload.get('composition')),
            lighting=_normalize_optional_text(payload.get('lighting')),
            raw_text=text,
            parsed=True,
        )

    def render_prompt(self) -> str:
        parts = [self.prompt]
        seen = {self.prompt.lower()}
        for item in (
            self.subject,
            self.action,
            self.scene,
            self.style,
            self.camera_motion,
            self.motion_level,
            self.composition,
            self.lighting,
        ):
            normalized = _normalize_text(item)
            if normalized and normalized.lower() not in seen:
                parts.append(normalized)
                seen.add(normalized.lower())
        return ', '.join(parts)

    def render_negative_prompt_additions(self) -> tuple[str, ...]:
        return _unique_normalized(self.negative_prompt_additions)

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {'prompt': self.prompt, 'parsed': self.parsed}
        if self.negative_prompt_additions:
            metadata['negative_prompt_additions'] = list(self.render_negative_prompt_additions())
        for key, value in (
            ('style', self.style),
            ('camera_motion', self.camera_motion),
            ('motion_level', self.motion_level),
            ('profile', self.profile),
            ('subject', self.subject),
            ('action', self.action),
            ('scene', self.scene),
            ('composition', self.composition),
            ('lighting', self.lighting),
        ):
            normalized = _normalize_text(value)
            if normalized:
                metadata[key] = normalized
        return metadata


def ar_stage_policy_for_pipeline(pipeline: str) -> ARStagePolicy:
    return MiniOmniPlanner().ar_stage_policy(pipeline)


def merge_negative_prompt(base_negative_prompt: str, additions: Iterable[str]) -> str:
    parts = []
    seen: set[str] = set()
    base = _normalize_text(base_negative_prompt)
    if base:
        parts.append(base)
        seen.add(base.lower())
    for item in additions:
        normalized = _normalize_text(item)
        if normalized and normalized.lower() not in seen:
            parts.append(normalized)
            seen.add(normalized.lower())
    return ', '.join(parts)


def _load_json_payload(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    if not candidate:
        return None
    if candidate.startswith('```'):
        lines = candidate.splitlines()
        if len(lines) >= 2:
            lines = lines[1:-1]
        candidate = '\n'.join(lines).strip()
        if candidate.startswith('json'):
            candidate = candidate[4:].strip()
    if candidate.startswith('{') and candidate.endswith('}'):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    start = candidate.find('{')
    end = candidate.rfind('}')
    if 0 <= start < end:
        try:
            payload = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _get_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return _normalize_text(value)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, (list, tuple, set)):
        return ', '.join(item for item in (_normalize_text(item) for item in value) if item)
    if not isinstance(value, str):
        value = str(value)
    return ' '.join(value.strip().split())


def _normalize_optional_text(value: Any) -> str | None:
    text = _normalize_text(value)
    return text or None


def _normalize_string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    return _unique_normalized(values)


def _unique_normalized(items: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = _normalize_text(item)
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(normalized)
    return tuple(result)
