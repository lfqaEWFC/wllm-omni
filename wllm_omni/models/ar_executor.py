from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from wllm_omni.model_types import ModelParadigm
from wllm_omni.models import ModelExecutor, supports_step_execution
from wllm_omni.models.ar_pipeline import ARPipeline, ARStepOutput, ARTextOutput, IdentityARPipeline
from wllm_omni.worker.utils import (
    ExecutionPhase,
    ExecutorCapability,
    ForwardBatch,
    ModelForwardOutput,
    RequestState,
    RunnerOutput,
)

if TYPE_CHECKING:
    from wllm_omni.request import OmniRequest

# The full stepwise contract, including finalize: the executor calls all
# three, so a pipeline offering only a subset must fall back to generate().
AR_STEP_EXECUTION_METHODS = ("prefill", "decode_step", "finalize")


@dataclass(slots=True)
class ARState:
    """Executor-side payload: the request plus opaque pipeline decode state.

    ``decode`` is whatever the pipeline's ``prefill``/``decode_step`` return;
    the executor only reads its ``finished`` attribute. It holds the KV cache,
    so it is dropped as soon as ``finalize`` has produced ``output``.
    """

    request: OmniRequest
    decode: Any = None
    output: ARTextOutput | None = None
    step_index: int = 0
    emitted_token_count: int = 0
    prefill_elapsed_s: float = 0.0
    decode_elapsed_s: float = 0.0
    prefill_steps: int = 0
    decode_steps: int = 0


class ARExecutor(ModelExecutor):
    """AR executor for the mini-Omni runtime.

    When the pipeline implements the full stepwise contract
    (``AR_STEP_EXECUTION_METHODS``, e.g. TransformersARPipeline), each
    ``forward()`` call advances exactly one step -- PREPARE runs prefill, STEP
    runs one KV-cache decode step -- mirroring DiffusionExecutor's
    PREPARE-then-STEP rhythm, except termination is decided by the pipeline
    (EOS / token budget) instead of a fixed step count. The decode state
    (including the KV cache) travels in ``ModelForwardOutput.payload`` and is
    written back in ``update_states``, so progress does not rely on in-process
    aliasing between the runner's state and the forward batch.

    Pipelines that only implement ``generate`` (e.g. IdentityARPipeline) run
    to completion in a single FINALIZE-phase forward call, unchanged.
    """

    paradigm = ModelParadigm.AUTOREGRESSIVE
    # Note: nothing reads .capabilities yet anywhere in the runtime; this set
    # is kept as-is from V0 rather than extended with more ornamental flags.
    capabilities = frozenset({ExecutorCapability.STEPWISE, ExecutorCapability.STREAMING})

    def __init__(self, pipeline: ARPipeline | None = None, *, emit_stream_events: bool = False):
        self.pipeline = pipeline or IdentityARPipeline()
        self.emit_stream_events = emit_stream_events
        self._stepwise = supports_step_execution(self.pipeline, AR_STEP_EXECUTION_METHODS)

    def init_state(self, sched_req_id: str, request: OmniRequest) -> RequestState:
        return RequestState(
            req_id=request.request_id,
            sched_req_id=sched_req_id,
            paradigm=self.paradigm,
            payload=ARState(request=request),
        )

    def batch_key(self, state: RequestState) -> tuple:
        # One request per forward batch: AR decode is per-sequence anyway
        # (the pipeline API takes a single request), and per-request batches
        # keep the batch phase well-defined -- same shape as DiffusionExecutor.
        return (self.paradigm.value, state.sched_req_id)

    def build_forward_batch(self, states: list[RequestState]) -> ForwardBatch:
        if len(states) != 1:
            raise ValueError(f"ARExecutor supports exactly one request per forward batch, got {len(states)}.")
        state = states[0]
        if not self._stepwise:
            # Monolithic pipelines produce the final output in one call.
            phase = ExecutionPhase.FINALIZE
        elif not state.initialized:
            phase = ExecutionPhase.PREPARE
        else:
            phase = ExecutionPhase.STEP
        return ForwardBatch(
            paradigm=self.paradigm,
            req_ids=[state.sched_req_id],
            phase=phase,
            payload=self._state_payload(state),
        )

    def forward(self, batch: ForwardBatch) -> ModelForwardOutput:
        if batch.paradigm != self.paradigm:
            raise ValueError(f"ARExecutor cannot run batch for paradigm={batch.paradigm}.")
        if len(batch.req_ids) != 1:
            raise ValueError(f"ARExecutor supports exactly one request per forward batch, got {len(batch.req_ids)}.")

        ar_state = self._batch_payload(batch)
        ar_state.step_index += 1
        events: list[ARStepOutput] = []

        if batch.phase == ExecutionPhase.FINALIZE:
            start = perf_counter()
            ar_state.output = self.pipeline.generate(ar_state.request)
            ar_state.decode_elapsed_s += perf_counter() - start
            events = self._final_output_stream_events(ar_state)
        elif batch.phase == ExecutionPhase.PREPARE:
            start = perf_counter()
            ar_state.decode = self.pipeline.prefill(ar_state.request)
            ar_state.prefill_elapsed_s += perf_counter() - start
            ar_state.prefill_steps += 1
            events = self._stream_events(ar_state)
        else:
            if ar_state.decode is None:
                raise ValueError("ARExecutor got a STEP batch without prefilled decode state.")
            start = perf_counter()
            ar_state.decode = self.pipeline.decode_step(ar_state.decode)
            ar_state.decode_elapsed_s += perf_counter() - start
            ar_state.decode_steps += 1
            events = self._stream_events(ar_state)

        if ar_state.output is None and ar_state.decode.finished:
            ar_state.output = self.pipeline.finalize(ar_state.decode)
            self._attach_runtime_metadata(ar_state)
            # The KV cache is dead weight once the output exists; drop it now
            # instead of keeping it alive until release().
            ar_state.decode = None
        elif ar_state.output is not None:
            self._attach_runtime_metadata(ar_state)

        finished = ar_state.output is not None
        runner_output = RunnerOutput(
            req_id=batch.req_ids[0],
            step_index=ar_state.step_index,
            finished=finished,
            result=ar_state.output if finished else None,
            events=events,
        )
        return ModelForwardOutput(outputs=[runner_output], payload=ar_state)

    # update_states / collect_outputs / release use the ModelExecutor defaults.

    def _stream_events(self, ar_state: ARState) -> list[ARStepOutput]:
        if not self.emit_stream_events or ar_state.decode is None:
            return []
        stream_events = getattr(self.pipeline, "stream_events", None)
        if not callable(stream_events):
            return []
        events = stream_events(ar_state.decode, ar_state.emitted_token_count)
        ar_state.emitted_token_count += len(events)
        return events

    def _final_output_stream_events(self, ar_state: ARState) -> list[ARStepOutput]:
        if not self.emit_stream_events or ar_state.output is None:
            return []
        token_id = ar_state.output.token_ids[0] if ar_state.output.token_ids else -1
        token = ar_state.output.tokens[0] if ar_state.output.tokens else ar_state.output.text
        ar_state.emitted_token_count = len(ar_state.output.token_ids)
        return [
            ARStepOutput(
                request_id=ar_state.output.request_id,
                token_index=0,
                token_id=token_id,
                token=token,
                text_delta=ar_state.output.text,
                finished=True,
                metadata={
                    "mode": ar_state.output.metadata.get("mode"),
                    "model": ar_state.output.metadata.get("model"),
                },
            )
        ]

    @staticmethod
    def _attach_runtime_metadata(ar_state: ARState) -> None:
        if ar_state.output is None:
            return
        total_elapsed_s = ar_state.prefill_elapsed_s + ar_state.decode_elapsed_s
        decode_step_mean_ms = (
            ar_state.decode_elapsed_s * 1000.0 / ar_state.decode_steps
            if ar_state.decode_steps > 0
            else 0.0
        )
        metadata = ar_state.output.metadata
        metadata.setdefault("streaming", False)
        metadata.setdefault("kv_cache_backend", metadata.get("kv_cache_type"))
        ar_state.output.metadata.update(
            scheduler_steps=ar_state.step_index,
            prefill_steps=ar_state.prefill_steps,
            decode_steps=ar_state.decode_steps,
            decode_model_calls=ar_state.decode_steps,
            decode_scheduler_steps=ar_state.decode_steps,
            prefill_ms=ar_state.prefill_elapsed_s * 1000.0,
            decode_ms=ar_state.decode_elapsed_s * 1000.0,
            elapsed_ms=total_elapsed_s * 1000.0,
            ttft_ms=ar_state.prefill_elapsed_s * 1000.0 if ar_state.prefill_steps else None,
            decode_step_mean_ms=decode_step_mean_ms,
        )

    @staticmethod
    def _state_payload(state: RequestState) -> ARState:
        if not isinstance(state.payload, ARState):
            raise TypeError(f"Expected ARState payload, got {type(state.payload).__name__}.")
        return state.payload

    @staticmethod
    def _batch_payload(batch: ForwardBatch) -> ARState:
        if not isinstance(batch.payload, ARState):
            raise TypeError(f"Expected ARState batch payload, got {type(batch.payload).__name__}.")
        return batch.payload
