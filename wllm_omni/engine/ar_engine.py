from __future__ import annotations

from typing import TYPE_CHECKING

from wllm_omni.config import EngineConfig
from wllm_omni.engine.planning import ar_stage_policy_for_pipeline
from wllm_omni.engine.model_runner import ModelRunner
from wllm_omni.model_types import ModelParadigm
from wllm_omni.models.ar_executor import ARExecutor
from wllm_omni.models.ar_pipeline import ARPipeline, ARStepOutput, ARTextOutput, TransformersARPipeline
from wllm_omni.sched.request_scheduler import RequestScheduler

if TYPE_CHECKING:
    from wllm_omni.request import OmniRequest


class AREngine:
    """Request-level AR engine for mini-Omni V0.

    V0 intentionally uses the generic RequestScheduler: AR-specific prefill /
    decode / KV-aware scheduling is not implemented yet, so exposing a named
    AR scheduler here would overstate the current runtime capability.
    """

    def __init__(self, config: EngineConfig, pipeline: ARPipeline | None = None):
        self.config = config
        self.scheduler = RequestScheduler(max_num_running_reqs=config.max_num_seqs)
        self.executor = ARExecutor(pipeline or self._make_pipeline(config))
        self.runner = ModelRunner(config, executors=[self.executor])
        self.last_output: ARTextOutput | None = None

    def generate(self, request: OmniRequest) -> ARTextOutput:
        request.model_paradigm = ModelParadigm.AUTOREGRESSIVE
        self.scheduler.add_request(request)
        outputs: list[ARTextOutput] = []
        while self.scheduler.has_requests():
            sched_output = self.scheduler.schedule()
            if sched_output.is_empty:
                break

            runner_output = self.runner.execute(sched_output)
            finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output)
            for finished_req_id in finished_req_ids:
                self.scheduler.pop_request_state(finished_req_id)

            for item in runner_output.outputs:
                if item.error is not None:
                    raise RuntimeError(item.error)
                if item.finished and isinstance(item.result, ARTextOutput):
                    outputs.append(item.result)

        if not outputs:
            raise RuntimeError("AR generation finished without output.")
        self.last_output = outputs[0]
        return outputs[0]

    def generate_stream(self, request: OmniRequest):
        request.model_paradigm = ModelParadigm.AUTOREGRESSIVE
        self.scheduler.add_request(request)
        outputs: list[ARTextOutput] = []
        self.executor.emit_stream_events = True
        try:
            while self.scheduler.has_requests():
                sched_output = self.scheduler.schedule()
                if sched_output.is_empty:
                    break

                runner_output = self.runner.execute(sched_output)
                finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output)
                for finished_req_id in finished_req_ids:
                    self.scheduler.pop_request_state(finished_req_id)

                for item in runner_output.outputs:
                    if item.error is not None:
                        raise RuntimeError(item.error)
                    for event in item.events:
                        if isinstance(event, ARStepOutput):
                            yield event
                    if item.finished and isinstance(item.result, ARTextOutput):
                        outputs.append(item.result)
        finally:
            self.executor.emit_stream_events = False

        if not outputs:
            raise RuntimeError("AR stream finished without output.")
        self.last_output = outputs[0]

    @staticmethod
    def _make_pipeline(config: EngineConfig) -> ARPipeline | None:
        if config.ar_model is None:
            return None
        policy = ar_stage_policy_for_pipeline(config.pipeline)
        return TransformersARPipeline(
            config.ar_model,
            device=config.device,
            dtype=config.dtype,
            local_files_only=config.local_files_only,
            token_budget=policy.token_budget,
            prompt_mode=policy.prompt_mode,
        )
