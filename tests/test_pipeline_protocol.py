from __future__ import annotations

import contextlib
import sys
import types

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    fake_torch = types.SimpleNamespace(
        bfloat16="bfloat16",
        float32="float32",
        dtype=object,
        no_grad=contextlib.nullcontext,
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    sys.modules["torch"] = fake_torch

try:
    from PIL import Image  # noqa: F401
except ModuleNotFoundError:
    fake_image_module = types.SimpleNamespace(Image=object)
    fake_pil = types.SimpleNamespace(Image=fake_image_module)
    sys.modules["PIL"] = fake_pil
    sys.modules["PIL.Image"] = fake_image_module

from wllm_omni.config import (
    PIPELINE_AR_TEXT,
    PIPELINE_QWEN_TO_WAN_I2V,
    PIPELINE_WAN_I2V,
    EngineConfig,
)
from wllm_omni.engine.ar_engine import AREngine
from wllm_omni.engine.mini_omni_runtime import MiniOmniRuntime
from wllm_omni.models.ar_pipeline import ARStepOutput, ARTextOutput, IdentityARPipeline
from wllm_omni.request import OmniRequest


class _StreamDecodeState:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.ids = [101]
        self.finished = False


class _StreamPipeline:
    total_tokens = 3

    def __init__(self):
        self.prefill_calls = 0
        self.decode_step_calls = 0

    def prefill(self, request: OmniRequest) -> _StreamDecodeState:
        self.prefill_calls += 1
        return _StreamDecodeState(request.request_id)

    def decode_step(self, state: _StreamDecodeState) -> _StreamDecodeState:
        self.decode_step_calls += 1
        state.ids.append(state.ids[-1] + 1)
        state.finished = len(state.ids) >= self.total_tokens
        return state

    def finalize(self, state: _StreamDecodeState) -> ARTextOutput:
        return ARTextOutput(
            request_id=state.request_id,
            text=" ".join(str(item) for item in state.ids),
            tokens=[str(item) for item in state.ids],
            token_ids=list(state.ids),
            metadata={
                "mode": "fake_stream",
                "model": "fake",
                "input_tokens": 1,
                "prefill_tokens": 1,
                "token_count": len(state.ids),
                "generated_tokens": len(state.ids),
                "stop_reason": "eos",
                "streaming": False,
                "kv_cache": True,
                "kv_cache_type": "FakeCache",
                "kv_cache_backend": "FakeCache",
                "kv_cache_source": "test",
                "runtime_kv_manager": False,
            },
        )

    def stream_events(self, state: _StreamDecodeState, emitted_token_count: int) -> list[ARStepOutput]:
        events = []
        for idx in range(emitted_token_count, len(state.ids)):
            token_id = state.ids[idx]
            events.append(
                ARStepOutput(
                    request_id=state.request_id,
                    token_index=idx,
                    token_id=token_id,
                    token=str(token_id),
                    text_delta=str(token_id),
                    finished=state.finished and idx == len(state.ids) - 1,
                )
            )
        return events

    def generate(self, request: OmniRequest) -> ARTextOutput:  # pragma: no cover
        raise AssertionError("stepwise pipeline should not call generate()")


def test_pipeline_graph_protocols_are_explicit():
    ar_runtime = MiniOmniRuntime(
        EngineConfig(enable_mini_omni=True, pipeline=PIPELINE_AR_TEXT),
        ar_pipeline=_StreamPipeline(),
    )
    assert ar_runtime.pipeline_registry.names == (
        PIPELINE_AR_TEXT,
        PIPELINE_WAN_I2V,
        PIPELINE_QWEN_TO_WAN_I2V,
    )
    assert list(ar_runtime.graph.nodes) == ["ar.text_generation"]
    assert ar_runtime.ar_stage is not None
    assert ar_runtime.diffusion_stage is None

    wan_runtime = MiniOmniRuntime(EngineConfig(enable_mini_omni=True, pipeline=PIPELINE_WAN_I2V))
    assert list(wan_runtime.graph.nodes) == ["diffusion.wan22_i2v"]
    assert wan_runtime.ar_stage is None
    assert wan_runtime.diffusion_stage is not None

    qwen_to_wan_runtime = MiniOmniRuntime(
        EngineConfig(enable_mini_omni=True, pipeline=PIPELINE_QWEN_TO_WAN_I2V),
        ar_pipeline=_StreamPipeline(),
    )
    assert list(qwen_to_wan_runtime.graph.nodes) == ["ar.prompt_bridge", "diffusion.wan22_i2v"]
    assert qwen_to_wan_runtime.ar_stage is not None
    assert qwen_to_wan_runtime.diffusion_stage is not None
    assert len(qwen_to_wan_runtime.graph.out_edges("ar.prompt_bridge")) == 1


def test_ar_engine_streams_step_events_through_runner():
    pipeline = _StreamPipeline()
    engine = AREngine(EngineConfig(enable_mini_omni=True, pipeline=PIPELINE_AR_TEXT), pipeline=pipeline)

    events = list(engine.generate_stream(OmniRequest(prompt="hello")))

    assert [event.token_id for event in events] == [101, 102, 103]
    assert events[-1].finished is True
    assert engine.last_output is not None
    assert engine.last_output.token_ids == [101, 102, 103]
    assert engine.last_output.metadata["scheduler_steps"] == 3
    assert engine.last_output.metadata["prefill_steps"] == 1
    assert engine.last_output.metadata["decode_model_calls"] == 2
    assert pipeline.prefill_calls == 1
    assert pipeline.decode_step_calls == 2


def test_ar_text_runtime_stream_trace_uses_pipeline_protocol():
    runtime = MiniOmniRuntime(
        EngineConfig(enable_mini_omni=True, pipeline=PIPELINE_AR_TEXT),
        ar_pipeline=_StreamPipeline(),
    )

    events = list(runtime.generate_ar_stream(OmniRequest(prompt="hello")))

    assert [event.text_delta for event in events] == ["101", "102", "103"]
    assert runtime.last_trace is not None
    assert runtime.last_trace.pipeline == PIPELINE_AR_TEXT
    assert runtime.last_trace.graph_nodes == ["ar.text_generation"]
    metadata = runtime.last_trace.stages[0].metadata
    assert metadata["streaming"] is True
    assert metadata["kv_cache_type"] == "FakeCache"
    assert metadata["kv_cache_backend"] == "FakeCache"
    assert metadata["kv_cache_source"] == "test"
    assert metadata["runtime_kv_manager"] is False


def test_ar_text_streaming_falls_back_for_single_shot_pipeline():
    runtime = MiniOmniRuntime(
        EngineConfig(enable_mini_omni=True, pipeline=PIPELINE_AR_TEXT),
        ar_pipeline=IdentityARPipeline(),
    )

    events = list(runtime.generate_ar_stream(OmniRequest(prompt="hello world")))

    assert len(events) == 1
    assert events[0].text_delta == "hello world"
    assert events[0].finished is True
    assert runtime.last_trace is not None
    metadata = runtime.last_trace.stages[0].metadata
    assert metadata["streaming"] is True
    assert metadata["kv_cache"] is False
