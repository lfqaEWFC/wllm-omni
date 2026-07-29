from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from wllm_omni.config import (
    AR_PROMPT_BRIDGE_NODE,
    DIFFUSION_WAN_I2V_NODE,
    PIPELINE_AR_TEXT,
    EngineConfig,
)
from wllm_omni.engine.connectors import ARToDiffusionConnector, CallableARToDiffusionConnector, StageConnector
from wllm_omni.engine.planning import MiniOmniPlanner, MiniOmniPipelinePlan
from wllm_omni.engine.stage import ARStage, DiffusionStage, StageOutput
from wllm_omni.engine.stage_graph import PipelineConfig, PipelineEdgeConfig, PipelineRegistry, StageGraph
from wllm_omni.engine.stage_scheduler import StageExecutionRecord, StageScheduler, StageSchedulerResult
from wllm_omni.model_types import ModelParadigm
from wllm_omni.models.ar_pipeline import ARPipeline, ARStepOutput, ARTextOutput
from wllm_omni.outputs import OmniOutput

if TYPE_CHECKING:
    from wllm_omni.request import OmniRequest


@dataclass(slots=True)
class OmniStageRecord:
    name: str
    paradigm: ModelParadigm
    request_id: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class MiniOmniTrace:
    request_id: str
    pipeline: str
    stages: list[OmniStageRecord] = field(default_factory=list)
    graph_nodes: list[str] = field(default_factory=list)


class MiniOmniRuntime:
    """A tiny vLLM-Omni-style stage-graph runtime for AR -> diffusion.

    The runtime owns the top-level stage graph. Each stage owns its own
    engine/scheduler/runner/executor stack.
    """

    def __init__(
        self,
        config: EngineConfig,
        connector: StageConnector | Callable[[OmniRequest, ARTextOutput], OmniRequest] | None = None,
        ar_pipeline: ARPipeline | None = None,
    ):
        self.config = config
        self.pipeline = config.pipeline
        self.planner = MiniOmniPlanner()
        self.pipeline_plan = self.planner.plan_pipeline(self.pipeline)
        self.pipeline_registry = self._build_pipeline_registry(self.planner)
        self.pipeline_config = self.pipeline_registry.get(self.pipeline)
        self.connector = self._normalize_connector(connector)
        self.ar_stage: ARStage | None = None
        self.diffusion_stage: DiffusionStage | None = None
        self.stage_nodes = self._build_stage_nodes(self.pipeline_config, ar_pipeline)
        self.graph = self._build_graph(self.pipeline)
        self.stage_scheduler = StageScheduler(self.graph)
        self.last_trace: MiniOmniTrace | None = None

    def generate_ar(self, request: OmniRequest) -> ARTextOutput:
        if self.pipeline != PIPELINE_AR_TEXT:
            raise RuntimeError("AR text generation requires pipeline='ar_text'.")
        result = self.stage_scheduler.run(request)
        self.last_trace = self._make_trace(result)
        if len(result.final_outputs) != 1:
            raise RuntimeError(f"AR text pipeline expected one output, got {len(result.final_outputs)}.")
        return self._ar_output(result.final_outputs[0])

    def generate_ar_stream(self, request: OmniRequest):
        self.planner.require_streamable(self.pipeline)
        for event in self.stage_scheduler.run_stream(request):
            if isinstance(event, ARStepOutput):
                yield event
        result = self.stage_scheduler.last_result
        if result is None:
            raise RuntimeError("AR stream finished without scheduler result.")
        if len(result.final_outputs) != 1:
            raise RuntimeError(f"AR stream pipeline expected one output, got {len(result.final_outputs)}.")
        self.last_trace = self._make_trace(result)

    def generate(self, request: OmniRequest) -> list[OmniOutput]:
        if self.pipeline == PIPELINE_AR_TEXT:
            raise RuntimeError("Use generate_ar() for pipeline='ar_text'.")
        result = self.stage_scheduler.run(request)
        self.last_trace = self._make_trace(result)
        return [self._diffusion_output(output) for output in result.final_outputs]

    def _build_graph(self, pipeline: str) -> StageGraph:
        return self.pipeline_registry.build_graph(
            pipeline,
            stages=self.stage_nodes,
            connectors={
                (AR_PROMPT_BRIDGE_NODE, DIFFUSION_WAN_I2V_NODE): self.connector,
            },
        )

    def _build_stage_nodes(
        self,
        pipeline_config: PipelineConfig,
        ar_pipeline: ARPipeline | None,
    ) -> dict[str, ARStage | DiffusionStage]:
        stages: dict[str, ARStage | DiffusionStage] = {}
        plan = self.planner.plan_pipeline(pipeline_config.name)
        for node_id in pipeline_config.nodes:
            if node_id in plan.nodes and plan.ar_policy is not None and node_id != DIFFUSION_WAN_I2V_NODE:
                self.ar_stage = ARStage(self.config, pipeline=ar_pipeline, name=node_id)
                stages[node_id] = self.ar_stage
            elif node_id == DIFFUSION_WAN_I2V_NODE:
                self.diffusion_stage = DiffusionStage(self.config)
                stages[node_id] = self.diffusion_stage
            else:
                raise ValueError(f"Unsupported stage node_id={node_id!r} in pipeline={pipeline_config.name!r}.")
        return stages

    @staticmethod
    def _build_pipeline_registry(planner: MiniOmniPlanner) -> PipelineRegistry:
        return PipelineRegistry([_pipeline_config_from_plan(plan) for plan in planner.plans])

    @staticmethod
    def _normalize_connector(
        connector: StageConnector | Callable[[OmniRequest, ARTextOutput], OmniRequest] | None,
    ) -> StageConnector:
        if connector is None:
            return ARToDiffusionConnector()
        if isinstance(connector, StageConnector):
            return connector
        return CallableARToDiffusionConnector(connector)

    def _make_trace(self, result: StageSchedulerResult) -> MiniOmniTrace:
        return MiniOmniTrace(
            request_id=result.root_request_id,
            pipeline=self.pipeline,
            stages=[self._make_stage_record_from_record(record) for record in result.records],
            graph_nodes=[record.node_id for record in result.records],
        )

    @staticmethod
    def _make_stage_record_from_record(record: StageExecutionRecord) -> OmniStageRecord:
        metadata = dict(record.metadata)
        paradigm = ModelParadigm(metadata.pop("paradigm"))
        return OmniStageRecord(
            name=record.stage_name,
            paradigm=paradigm,
            request_id=record.request_id,
            metadata=metadata,
        )

    @staticmethod
    def _ar_output(output: StageOutput) -> ARTextOutput:
        if not isinstance(output.data, ARTextOutput):
            raise TypeError(f"Expected ARTextOutput, got {type(output.data).__name__}.")
        return output.data

    @staticmethod
    def _diffusion_output(output: StageOutput) -> OmniOutput:
        if not isinstance(output.data, OmniOutput):
            raise TypeError(f"Expected OmniOutput, got {type(output.data).__name__}.")
        return output.data


def _pipeline_config_from_plan(plan: MiniOmniPipelinePlan) -> PipelineConfig:
    return PipelineConfig(
        name=plan.name,
        nodes=plan.nodes,
        edges=tuple(PipelineEdgeConfig(source, target) for source, target in plan.edges),
    )
