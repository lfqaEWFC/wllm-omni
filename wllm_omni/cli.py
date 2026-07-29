from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - lightweight AR-only CLI import path
    torch = None

from wllm_omni import DEFAULT_IMAGE, DEFAULT_MODEL, DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT, OmniLLM
from wllm_omni.config import PIPELINE_AR_TEXT, PIPELINE_QWEN_TO_WAN_I2V, PIPELINE_WAN_I2V, SUPPORTED_PIPELINES


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal mini-Omni pipeline runner.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--preset", choices=["quality"], default="quality")
    parser.add_argument("--output", default="./download/example_wan22_i2v_quality.mp4")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--disable-cpu-offload", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--profile", action="store_true", help="Print a per-request diffusion profiler summary.")
    parser.add_argument(
        "--vae-dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="VAE load/decode dtype. fp32 is the stable default; bf16 can be profiled as an experimental speed policy.",
    )
    parser.add_argument(
        "--probe-condition-cache",
        action="store_true",
        help="Run an extra prepare_latents probe to check whether Wan condition tensors are seed-independent.",
    )
    parser.add_argument(
        "--pipeline",
        choices=SUPPORTED_PIPELINES,
        default=PIPELINE_WAN_I2V,
        help="Explicit mini-Omni pipeline: ar_text, wan_i2v, or qwen_to_wan_i2v.",
    )
    parser.add_argument(
        "--ar-model",
        default=None,
        help="Local CausalLM model path for AR pipelines.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream AR text deltas. Supported only with --pipeline ar_text.",
    )
    return parser.parse_args()


def _parse_vae_dtype(value: str):
    if torch is None:
        raise RuntimeError("PyTorch is required when --vae-dtype is used for diffusion execution.")
    if value == "fp32":
        return torch.float32
    if value == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported VAE dtype: {value}")


def _format_elapsed_ms(elapsed_s) -> str:
    if elapsed_s is None:
        return "None"
    return f"{float(elapsed_s) * 1000.0:.2f}"


def _format_ms(value) -> str:
    if value is None:
        return "None"
    return f"{float(value):.2f}"


def _format_text(value) -> str:
    if value is None:
        return "None"
    return json.dumps(str(value), ensure_ascii=False)


def _print_mini_omni_trace(trace) -> None:
    if trace is None:
        return
    stages = " -> ".join(stage.name for stage in trace.stages)
    graph_nodes = " -> ".join(getattr(trace, "graph_nodes", []) or [])
    pipeline = getattr(trace, "pipeline", None)
    print(f"[wllm-omni][mini-omni] request_id={trace.request_id} pipeline={pipeline} stages={stages}", flush=True)
    if graph_nodes:
        print(f"[wllm-omni][mini-omni] graph={graph_nodes}", flush=True)
    for stage in trace.stages:
        metadata = stage.metadata
        elapsed_text = _format_elapsed_ms(metadata.get("elapsed_s"))
        if getattr(stage.paradigm, "value", stage.paradigm) == "ar":
            print(
                "[wllm-omni][mini-omni] "
                f"ar.model={metadata.get('model')} "
                f"ar.mode={metadata.get('mode')} "
                f"ar.prompt_mode={metadata.get('prompt_mode')} "
                f"ar.input_tokens={metadata.get('input_tokens')} "
                f"ar.prefill_tokens={metadata.get('prefill_tokens')} "
                f"ar.output_tokens={metadata.get('output_tokens')} "
                f"ar.generated_tokens={metadata.get('generated_tokens')} "
                f"ar.elapsed_ms={_format_ms(metadata.get('elapsed_ms')) if metadata.get('elapsed_ms') is not None else elapsed_text} "
                f"ar.prefill_ms={_format_ms(metadata.get('prefill_ms'))} "
                f"ar.decode_ms={_format_ms(metadata.get('decode_ms'))} "
                f"ar.ttft_ms={_format_ms(metadata.get('ttft_ms'))} "
                f"ar.scheduler_steps={metadata.get('scheduler_steps')} "
                f"ar.prefill_steps={metadata.get('prefill_steps')} "
                f"ar.decode_model_calls={metadata.get('decode_model_calls')} "
                f"ar.decode_scheduler_steps={metadata.get('decode_scheduler_steps')} "
                f"ar.decode_step_mean_ms={_format_ms(metadata.get('decode_step_mean_ms'))} "
                f"ar.stop_reason={metadata.get('stop_reason')} "
                f"ar.streaming={metadata.get('streaming')} "
                f"ar.kv_cache={metadata.get('kv_cache')} "
                f"ar.kv_cache_type={metadata.get('kv_cache_type')} "
                f"ar.kv_cache_backend={metadata.get('kv_cache_backend')} "
                f"ar.kv_cache_source={metadata.get('kv_cache_source')} "
                f"ar.runtime_kv_manager={metadata.get('runtime_kv_manager')}",
                flush=True,
            )
            if stage.name == "ar.prompt_bridge":
                print(
                    "[wllm-omni][mini-omni] "
                    f"ar.output_text={_format_text(metadata.get('output_text'))}",
                    flush=True,
                )
        elif stage.name == "diffusion.wan22_i2v":
            load_elapsed_text = _format_elapsed_ms(metadata.get("load_elapsed_s"))
            request_extra = metadata.get("request_extra")
            diffusion_prompt = request_extra.get("diffusion_prompt") if isinstance(request_extra, dict) else None
            bridge_parse_success = request_extra.get("bridge_parse_success") if isinstance(request_extra, dict) else None
            bridge_fallback = request_extra.get("bridge_fallback") if isinstance(request_extra, dict) else None
            bridge_fallback_reason = request_extra.get("bridge_fallback_reason") if isinstance(request_extra, dict) else None
            bridge_strategy = request_extra.get("bridge_strategy") if isinstance(request_extra, dict) else None
            ar_guidance_prompt = request_extra.get("ar_guidance_prompt") if isinstance(request_extra, dict) else None
            print(
                "[wllm-omni][mini-omni] "
                f"diffusion.bridge={metadata.get('bridge')} "
                f"diffusion.source_node={metadata.get('source_node')} "
                f"diffusion.source_request_id={metadata.get('source_request_id')} "
                f"diffusion.load_was_cold={metadata.get('load_was_cold')} "
                f"diffusion.load_ms={load_elapsed_text} "
                f"diffusion.elapsed_ms={elapsed_text} "
                f"diffusion.bridge_strategy={bridge_strategy} "
                f"diffusion.bridge_parse_success={bridge_parse_success} "
                f"diffusion.bridge_fallback={bridge_fallback} "
                f"diffusion.bridge_fallback_reason={bridge_fallback_reason}",
                flush=True,
            )
            if ar_guidance_prompt is not None:
                print(
                    "[wllm-omni][mini-omni] "
                    f"diffusion.ar_guidance_prompt={_format_text(ar_guidance_prompt)}",
                    flush=True,
                )
            if diffusion_prompt is not None:
                print(
                    "[wllm-omni][mini-omni] "
                    f"diffusion.prompt={_format_text(diffusion_prompt)}",
                    flush=True,
                )


def _validate_args(args) -> None:
    ar_pipelines = {PIPELINE_AR_TEXT, PIPELINE_QWEN_TO_WAN_I2V}
    if args.ar_model is not None and args.pipeline not in ar_pipelines:
        raise ValueError("--ar-model is only valid with --pipeline ar_text or qwen_to_wan_i2v.")
    if args.stream and args.pipeline != PIPELINE_AR_TEXT:
        raise ValueError("--stream is only supported with --pipeline ar_text.")


def main():
    args = parse_args()
    _validate_args(args)
    config_kwargs = {
        "use_cpu_offload": not args.disable_cpu_offload,
        "max_num_seqs": args.max_num_seqs,
        "enable_profiling": args.profile,
        "probe_condition_cache": args.probe_condition_cache,
        "enable_mini_omni": True,
        "pipeline": args.pipeline,
        "ar_model": args.ar_model,
    }
    if args.pipeline != PIPELINE_AR_TEXT or torch is not None:
        config_kwargs["vae_dtype"] = _parse_vae_dtype(args.vae_dtype)
    llm = OmniLLM(args.model, **config_kwargs)
    sampling_params = llm.preset(args.preset)
    if args.seed is not None:
        sampling_params.seed = args.seed

    if args.pipeline == PIPELINE_AR_TEXT:
        if args.stream:
            for chunk in llm.generate_ar_stream(args.prompt):
                print(chunk.text_delta, end="", flush=True)
            print(flush=True)
            _print_mini_omni_trace(llm.last_omni_trace)
            return
        ar_output = llm.generate_ar(args.prompt)
        _print_mini_omni_trace(llm.last_omni_trace)
        print(ar_output.text)
        return

    generation = llm.generate(args.image, args.prompt, args.negative_prompt, sampling_params)
    output_path = Path(args.output)
    llm.save(output=generation, output_path=output_path)
    _print_mini_omni_trace(llm.last_omni_trace)
    print(output_path.resolve())
