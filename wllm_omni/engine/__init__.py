from __future__ import annotations

__all__ = [
    "DiffusionEngine",
    "ModelRunner",
    "PipelineConfig",
    "PipelineEdgeConfig",
    "PipelineRegistry",
    "StageGraph",
    "StageScheduler",
]


def __getattr__(name: str):
    if name == "DiffusionEngine":
        from wllm_omni.engine.diffusion_engine import DiffusionEngine

        return DiffusionEngine
    if name == "ModelRunner":
        from wllm_omni.engine.model_runner import ModelRunner

        return ModelRunner
    if name in {"PipelineConfig", "PipelineEdgeConfig", "PipelineRegistry", "StageGraph"}:
        from wllm_omni.engine.stage_graph import PipelineConfig, PipelineEdgeConfig, PipelineRegistry, StageGraph

        return {
            "PipelineConfig": PipelineConfig,
            "PipelineEdgeConfig": PipelineEdgeConfig,
            "PipelineRegistry": PipelineRegistry,
            "StageGraph": StageGraph,
        }[name]
    if name == "StageScheduler":
        from wllm_omni.engine.stage_scheduler import StageScheduler

        return StageScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
