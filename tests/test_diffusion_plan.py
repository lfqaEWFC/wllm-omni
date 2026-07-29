from __future__ import annotations

from wllm_omni.config import (
    AR_PROMPT_BRIDGE_NODE,
    AR_TEXT_NODE,
    DIFFUSION_WAN_I2V_NODE,
    PIPELINE_AR_TEXT,
    PIPELINE_QWEN_TO_WAN_I2V,
    PIPELINE_WAN_I2V,
)
from wllm_omni.engine.connectors import ARToDiffusionConnector, ConnectorContext
from wllm_omni.engine.planning import DiffusionPlan, MiniOmniPlanner, ar_stage_policy_for_pipeline
from wllm_omni.engine.stage import StageOutput
from wllm_omni.model_types import ModelParadigm
from wllm_omni.models.ar_pipeline import ARTextOutput
from wllm_omni.request import OmniRequest
from wllm_omni.sampling_params import OmniSamplingParams


def test_diffusion_plan_parses_json_payload_and_renders_prompt():
    plan = DiffusionPlan.from_text(
        '{"prompt":"a cat on a surfboard","negative_prompt_additions":["static","blur"],'
        '"style":"cinematic","camera_motion":"slow_push_in","motion_level":"medium","profile":"quality"}'
    )

    assert plan.prompt == 'a cat on a surfboard'
    assert plan.negative_prompt_additions == ('static', 'blur')
    assert 'cinematic' in plan.render_prompt()
    assert plan.to_metadata()['profile'] == 'quality'


def test_ar_to_diffusion_connector_keeps_shape_params_and_attaches_plan_metadata():
    sampling = OmniSamplingParams(
        height=800,
        width=576,
        num_frames=17,
        num_inference_steps=12,
        guidance_scale=5.0,
        flow_shift=3.0,
        negative_prompt='base negative',
        seed=42,
        fps=16,
    )
    root_request = OmniRequest(
        prompt='original prompt',
        image='image.png',
        sampling_params=sampling,
        model_paradigm=ModelParadigm.DIFFUSION,
        request_id='req-1',
    )
    ar_output = ARTextOutput(
        request_id='req-1',
        text='{"prompt":"a cat on a surfboard","negative_prompt_additions":["static"],"style":"cinematic"}',
        tokens=['a'],
        token_ids=[1],
    )
    context = ConnectorContext(
        root_request=root_request,
        source_node='ar.prompt_bridge',
        target_node='diffusion.wan22_i2v',
        source_output=StageOutput(request_id='req-1', data=ar_output, metadata={}),
    )

    request = ARToDiffusionConnector().connect(context)

    assert request.prompt == (
        'original prompt Additional motion/camera guidance from AR: '
        'a cat on a surfboard, cinematic.'
    )
    assert request.sampling_params.height == 800
    assert request.sampling_params.width == 576
    assert request.sampling_params.num_frames == 17
    assert request.sampling_params.negative_prompt == 'base negative, static'
    assert request.extra['bridge'] == 'ar_text_to_diffusion_prompt'
    assert request.extra['bridge_strategy'] == 'preserve_root_prompt_with_ar_guidance'
    assert request.extra['ar_guidance_prompt'] == 'a cat on a surfboard, cinematic'
    assert request.extra['diffusion_plan']['prompt'] == 'a cat on a surfboard'
    assert request.extra['diffusion_plan']['style'] == 'cinematic'
    assert root_request.sampling_params.negative_prompt == 'base negative'


def test_ar_to_diffusion_connector_accepts_plain_prompt_output():
    sampling = OmniSamplingParams(800, 576, 17, 12, 5.0, 3.0, negative_prompt='base negative')
    root_request = OmniRequest(
        prompt='original dog prompt',
        image='image.png',
        sampling_params=sampling,
        model_paradigm=ModelParadigm.DIFFUSION,
        request_id='req-plain',
    )
    ar_output = ARTextOutput(
        request_id='req-plain',
        text='golden retriever puppy blinks, tilts its head, and pants on a green lawn',
        tokens=['golden'],
        token_ids=[1],
    )
    context = ConnectorContext(
        root_request=root_request,
        source_node='ar.prompt_bridge',
        target_node='diffusion.wan22_i2v',
        source_output=StageOutput(request_id='req-plain', data=ar_output, metadata={}),
    )

    request = ARToDiffusionConnector().connect(context)

    assert request.prompt == (
        'original dog prompt Additional motion/camera guidance from AR: '
        'golden retriever puppy blinks, tilts its head, and pants on a green lawn.'
    )
    assert request.extra['ar_guidance_prompt'] == (
        'golden retriever puppy blinks, tilts its head, and pants on a green lawn'
    )
    assert request.extra['bridge_parse_success'] is False
    assert request.extra['bridge_fallback'] is False


def test_ar_to_diffusion_connector_falls_back_on_truncated_ar_guidance():
    sampling = OmniSamplingParams(800, 576, 17, 12, 5.0, 3.0, negative_prompt='base negative')
    root_request = OmniRequest(
        prompt='original dog prompt with blinking and head tilt',
        image='image.png',
        sampling_params=sampling,
        model_paradigm=ModelParadigm.DIFFUSION,
        request_id='req-truncated',
    )
    ar_output = ARTextOutput(
        request_id='req-truncated',
        text='gentle blinking, playful panting, stable camera, friendly and',
        tokens=['bad'],
        token_ids=[1],
        metadata={'stop_reason': 'token_budget'},
    )
    context = ConnectorContext(
        root_request=root_request,
        source_node='ar.prompt_bridge',
        target_node='diffusion.wan22_i2v',
        source_output=StageOutput(request_id='req-truncated', data=ar_output, metadata={}),
    )

    request = ARToDiffusionConnector().connect(context)

    assert request.prompt == 'original dog prompt with blinking and head tilt'
    assert request.extra['ar_guidance_prompt'] == ''
    assert request.extra['bridge_fallback'] is True
    assert request.extra['bridge_fallback_reason'] == 'truncated_ar_guidance'


def test_ar_to_diffusion_connector_falls_back_on_broken_structured_output():
    sampling = OmniSamplingParams(800, 576, 17, 12, 5.0, 3.0, negative_prompt='base negative')
    root_request = OmniRequest(
        prompt='original dog prompt with blinking and head tilt',
        image='image.png',
        sampling_params=sampling,
        model_paradigm=ModelParadigm.DIFFUSION,
        request_id='req-broken',
    )
    ar_output = ARTextOutput(
        request_id='req-broken',
        text='```json\n{\n  "prompt": "truncated puppy prompt",\n  "camera_motion": ["soft',
        tokens=['bad'],
        token_ids=[1],
    )
    context = ConnectorContext(
        root_request=root_request,
        source_node='ar.prompt_bridge',
        target_node='diffusion.wan22_i2v',
        source_output=StageOutput(request_id='req-broken', data=ar_output, metadata={}),
    )

    request = ARToDiffusionConnector().connect(context)

    assert request.prompt == 'original dog prompt with blinking and head tilt'
    assert request.extra['bridge_parse_success'] is False
    assert request.extra['bridge_fallback'] is True
    assert request.extra['bridge_fallback_reason'] == 'invalid_structured_ar_output'


def test_ar_stage_policy_limits_only_qwen_to_wan_bridge():
    text_policy = ar_stage_policy_for_pipeline(PIPELINE_AR_TEXT)
    assert text_policy.role == 'text_generation'
    assert text_policy.output_contract == 'final_text'
    assert text_policy.allow_stream is True
    assert text_policy.token_budget is None

    bridge_policy = ar_stage_policy_for_pipeline(PIPELINE_QWEN_TO_WAN_I2V)
    assert bridge_policy.role == 'prompt_bridge'
    assert bridge_policy.output_contract == 'supplemental_guidance'
    assert bridge_policy.allow_stream is False
    assert bridge_policy.token_budget == 64


def test_mini_omni_planner_returns_static_pipeline_plans():
    planner = MiniOmniPlanner()

    ar_plan = planner.plan_pipeline(PIPELINE_AR_TEXT)
    assert ar_plan.nodes == (AR_TEXT_NODE,)
    assert ar_plan.edges == ()
    assert ar_plan.allow_stream is True
    assert ar_plan.requires_ar is True
    assert ar_plan.requires_diffusion is False

    wan_plan = planner.plan_pipeline(PIPELINE_WAN_I2V)
    assert wan_plan.nodes == (DIFFUSION_WAN_I2V_NODE,)
    assert wan_plan.edges == ()
    assert wan_plan.allow_stream is False
    assert wan_plan.requires_ar is False
    assert wan_plan.requires_diffusion is True

    bridge_plan = planner.plan_pipeline(PIPELINE_QWEN_TO_WAN_I2V)
    assert bridge_plan.nodes == (AR_PROMPT_BRIDGE_NODE, DIFFUSION_WAN_I2V_NODE)
    assert bridge_plan.edges == ((AR_PROMPT_BRIDGE_NODE, DIFFUSION_WAN_I2V_NODE),)
    assert bridge_plan.allow_stream is False
    assert bridge_plan.requires_ar is True
    assert bridge_plan.requires_diffusion is True
