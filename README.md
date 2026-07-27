# wllm-omni

`wllm-omni` 是一个用于学习 AI Infra / 多模态推理框架的轻量项目。当前目标是手写一个 mini vLLM-Omni 风格的单进程 runtime：用显式 pipeline / stage graph 组织不同模型范式，并让每个 stage 继续走自己的 scheduler、runner 和 executor。

## 当前定位

当前 V0 已支持：

- Wan2.2 image-to-video diffusion 执行
- Wan profiler、VAE dtype policy、prepare-stage cache
- Qwen / Transformers CausalLM AR stage
- AR prefill / decode step / finalize 拆分
- AR 当前使用通用 `RequestScheduler`，尚未实现专用 token / KV-aware 调度
- Transformers `past_key_values` KV cache 透传
- AR token streaming（`ar_text` pipeline）
- 显式 pipeline 协议：`ar_text`、`wan_i2v`、`qwen_to_wan_i2v`
- `ar_text` 使用 `ar.text_generation` 节点，`qwen_to_wan_i2v` 使用 `ar.prompt_bridge` 节点
- `PipelineRegistry` 管理已配置 stage graph，避免 runtime 内硬编码 pipeline 分支
- `StageGraph -> StageScheduler -> Stage -> Engine -> Scheduler -> ModelRunner -> Executor` 分层

还没有支持：

- runtime 自己管理的 paged KV / KV block manager
- AR decode batching
- 多 session 调度
- stage-level batching
- pipeline overlap
- 多 GPU / 分布式 stage serving

## 架构

```text
MiniOmniRuntime
  ├── PipelineRegistry
  │     ├── ar_text: ar.text_generation
  │     ├── wan_i2v: diffusion.wan22_i2v
  │     └── qwen_to_wan_i2v: ar.prompt_bridge -> diffusion.wan22_i2v
  │
  └── StageScheduler
        └── StageGraph(selected pipeline)
              ├── ARStage(ar.text_generation / ar.prompt_bridge)
              │     └── AREngine
              │           └── RequestScheduler
              │                 └── ModelRunner
              │                       └── ARExecutor
              │                             └── TransformersARPipeline / IdentityARPipeline
              │
              ├── Connector(qwen_to_wan_i2v only)
              │     AR text output + image + sampling params
              │     -> diffusion OmniRequest
              │
              └── DiffusionStage
                    └── DiffusionEngine
                          └── StepScheduler
                                └── ModelRunner
                                      └── DiffusionExecutor
                                            └── Wan22I2VPipeline
```

核心原则：

- 顶层用 `--pipeline` 选择 `PipelineRegistry` 中已配置好的 stage graph
- `ModelRunner` 保持通用
- AR / Diffusion 差异下沉到 executor 和 pipeline
- AR streaming 不绕过 runner，而是通过 `RunnerOutput.events` 输出 token delta

## 安装

```text
conda create -n wllm-omni python=3.11 -y
conda activate wllm-omni
```

按适配的 CUDA 版本安装 PyTorch，例如 CUDA 12.1：

```text
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```text
python -m pip install -e .
python -m pip install huggingface_hub
```

## 模型下载

Wan diffusion 模型：

```text
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --local-dir ./models/Wan2.2-TI2V-5B-Diffusers
```

Qwen AR 模型：

```text
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir ./models/Qwen2.5-0.5B-Instruct
```

## 运行命令

### AR text

`ar_text` 是普通文本生成 pipeline，graph 节点是 `ar.text_generation`，不会套 image-to-video prompt rewrite 模板。

```text
python example_wan22_i2v.py \
  --pipeline ar_text \
  --ar-model ./models/Qwen2.5-0.5B-Instruct \
  --ar-max-new-tokens 1000 \
  --prompt "生成1000字的文本，描述一所学校"
```

### AR streaming

```text
python example_wan22_i2v.py \
  --pipeline ar_text \
  --ar-model ./models/Qwen2.5-0.5B-Instruct \
  --stream \
  --prompt "A cat wearing sunglasses sits on a surfboard at the beach."
```

### Wan I2V

```text
unset OMP_NUM_THREADS

CUDA_VISIBLE_DEVICES=0 python example_wan22_i2v.py \
  --model ./models/Wan2.2-TI2V-5B-Diffusers \
  --image ./assets/image.png \
  --preset quality \
  --output ./output/pipeline_wan_i2v.mp4 \
  --profile \
  --disable-cpu-offload \
  --vae-dtype bf16 \
  --pipeline wan_i2v
```

### Qwen -> Wan I2V

`qwen_to_wan_i2v` 才会启用 AR prompt bridge，graph 节点是 `ar.prompt_bridge -> diffusion.wan22_i2v`：先把用户输入改写成适合 Wan I2V 的视觉 prompt，再交给 diffusion stage。

```text
unset OMP_NUM_THREADS

CUDA_VISIBLE_DEVICES=0 python example_wan22_i2v.py \
  --model ./models/Wan2.2-TI2V-5B-Diffusers \
  --image ./assets/image.png \
  --preset quality \
  --output ./output/pipeline_qwen_to_wan.mp4 \
  --profile \
  --disable-cpu-offload \
  --vae-dtype bf16 \
  --pipeline qwen_to_wan_i2v \
  --ar-model ./models/Qwen2.5-0.5B-Instruct \
  --ar-max-new-tokens 64
```

## Trace 字段

AR stage 会输出：

```text
ar.input_tokens
ar.prefill_tokens
ar.output_tokens
ar.generated_tokens
ar.prefill_ms
ar.decode_ms
ar.ttft_ms
ar.scheduler_steps
ar.prefill_steps
ar.decode_model_calls
ar.decode_scheduler_steps
ar.decode_step_mean_ms
ar.stop_reason
ar.streaming
ar.kv_cache
ar.kv_cache_type
ar.kv_cache_backend
ar.kv_cache_source
ar.runtime_kv_manager
```

Diffusion stage 会输出：

```text
diffusion.bridge
diffusion.source_node
diffusion.source_request_id
diffusion.load_was_cold
diffusion.load_ms
diffusion.elapsed_ms
```

## KV Cache 边界

当前 AR KV cache 来自 Transformers CausalLM 的 `past_key_values`，由 `TransformersARPipeline.prefill()` 得到，并在 `decode_step()` 中继续传入模型。`ar.prompt_mode=text` 表示普通文本生成，`ar.prompt_mode=i2v_bridge` 表示 AR 输出会作为 diffusion prompt bridge。

当前还不是 vLLM 式 KV cache manager：

- 没有 paged KV block
- 没有 block table
- 没有 prefix cache
- 没有 eviction / migration
- 没有多请求 decode batching

因此 trace 中会标记：

```text
ar.kv_cache_backend=DynamicCache
ar.kv_cache_source=transformers_past_key_values
ar.runtime_kv_manager=False
```

## 测试

轻量协议测试：

```text
python -m pytest tests/test_pipeline_protocol.py -q
```

AR stepwise / generate fidelity 测试：

```text
python -m pytest tests/test_ar_stepwise.py -q
```

如果本地没有 `pytest`，可先安装：

```text
python -m pip install pytest
```

## 下一步

建议下一步继续做 AR 侧 runtime 优化：

1. 先设计真正的 AR scheduler 语义：prefill queue、decode queue、token budget
2. 再引入 AR KV metadata / KV block manager 雏形
3. 支持多请求 decode batching
4. 继续推进 prefix cache、stream server、多 session 调度
5. 最后再考虑 AR-Diffusion pipeline overlap
