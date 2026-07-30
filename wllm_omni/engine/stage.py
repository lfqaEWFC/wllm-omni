from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any

from wllm_omni.config import EngineConfig
from wllm_omni.engine.ar_engine import AREngine
from wllm_omni.engine.diffusion_engine import DiffusionEngine
from wllm_omni.model_types import ModelParadigm
from wllm_omni.models.ar_pipeline import ARPipeline, ARTextOutput
from wllm_omni.outputs import OmniOutput

if TYPE_CHECKING:
    from wllm_omni.request import OmniRequest


@dataclass(slots=True)
class StageOutput:
    request_id: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    name: str
    paradigm: ModelParadigm

    def prepare(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def run(self, request: OmniRequest) -> StageOutput:
        pass

    def run_stream(self, request: OmniRequest):
        if False:
            yield None
        return self.run(request)


class ARStage(Stage):
    name = "ar.text_generation"
    paradigm = ModelParadigm.AUTOREGRESSIVE

    def __init__(self, config: EngineConfig, pipeline: ARPipeline | None = None, *, name: str | None = None):
        if name is not None:
            self.name = name
        self.engine = AREngine(config, pipeline=pipeline)

    def run(self, request: OmniRequest) -> StageOutput:
        ar_output = self.engine.generate(request)
        return StageOutput(
            request_id=request.request_id,
            data=ar_output,
            metadata=self.metadata_from_output(ar_output),
        )

    def run_stream(self, request: OmniRequest):
        for event in self.engine.generate_stream(request):
            yield event
        ar_output = self.engine.last_output
        if ar_output is None:
            raise RuntimeError("AR stream finished without output.")
        ar_output.metadata["streaming"] = True
        return StageOutput(
            request_id=request.request_id,
            data=ar_output,
            metadata=self.metadata_from_output(ar_output),
        )

    @staticmethod
    def metadata_from_output(ar_output: ARTextOutput) -> dict[str, Any]:
        metadata = dict(ar_output.metadata)
        metadata.setdefault("output_tokens", metadata.get("token_count", len(ar_output.tokens)))
        metadata.setdefault("output_text", ar_output.text)
        return metadata


class DiffusionStage(Stage):
    name = "diffusion.wan22_i2v"
    paradigm = ModelParadigm.DIFFUSION

    def __init__(self, config: EngineConfig):
        self.config = config
        self.engine: DiffusionEngine | None = None

    def prepare(self) -> dict[str, Any]:
        if self.engine is not None:
            return {"load_elapsed_s": 0.0, "load_was_cold": False}
        start = perf_counter()
        self.engine = DiffusionEngine(self.config)
        return {"load_elapsed_s": perf_counter() - start, "load_was_cold": True}

    def run(self, request: OmniRequest) -> StageOutput:
        outputs = self._engine().generate(request)
        if not outputs:
            raise RuntimeError("Diffusion stage finished without output.")
        return StageOutput(
            request_id=request.request_id,
            data=outputs[0],
            metadata={},
        )

    def _engine(self) -> DiffusionEngine:
        if self.engine is None:
            self.engine = DiffusionEngine(self.config)
        return self.engine
