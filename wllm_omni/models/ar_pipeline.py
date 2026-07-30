from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
from typing import TYPE_CHECKING, Any

from wllm_omni.config import AR_PROMPT_MODE_I2V_BRIDGE, AR_PROMPT_MODE_TEXT, SUPPORTED_AR_PROMPT_MODES

if TYPE_CHECKING:
    from wllm_omni.request import OmniRequest


@dataclass(slots=True)
class ARTextOutput:
    request_id: str
    text: str
    tokens: list[str]
    token_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ARStepOutput:
    request_id: str
    token_index: int
    token_id: int
    token: str
    text_delta: str
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ARDecodeState:
    """Per-request state for stepwise (prefill + KV-cache decode) generation.

    Produced by ``ARPipeline.prefill`` and threaded through ``decode_step``
    until ``finished`` is True, then turned into an ``ARTextOutput`` by
    ``finalize``. Executors only ever read ``finished``; every other field is
    owned by the pipeline that produced the state. For
    ``TransformersARPipeline`` the ``Any`` fields hold torch tensors, a
    transformers KV ``Cache``, and the ``LogitsProcessorList`` /
    ``StoppingCriteriaList`` built by transformers' own generation machinery
    at prefill time (built once, reused every step).
    """

    request_id: str
    prompt: str
    input_length: int
    input_ids: Any
    attention_mask: Any
    logits_processor: Any
    stopping_criteria: Any
    cache: Any = None
    token_budget: int | None = None
    context_window: int | None = None
    finished: bool = False


class ARPipeline(ABC):
    """Text-generation stage interface for mini-Omni composition.

    ``generate`` is the only required method: run one request to completion.
    Pipelines that can decode incrementally may additionally implement the
    stepwise contract -- ``prefill`` / ``decode_step`` / ``finalize`` (see
    ``wllm_omni.models.ar_executor.AR_STEP_EXECUTION_METHODS``) -- so that
    ARExecutor can advance one decode step per ``forward()`` call instead of
    blocking on a single monolithic call.
    """

    @abstractmethod
    def generate(self, request: OmniRequest) -> ARTextOutput:
        pass

    def stream_events(self, state: Any, emitted_token_count: int) -> list[ARStepOutput]:
        return []


class IdentityARPipeline(ARPipeline):
    """Deterministic AR placeholder used before a real causal LM backend."""

    def generate(self, request: OmniRequest) -> ARTextOutput:
        text = self._normalize_prompt(request.prompt)
        tokens = self._tokenize(text)
        return ARTextOutput(
            request_id=request.request_id,
            text=text,
            tokens=tokens,
            token_ids=[self._stable_token_id(token) for token in tokens],
            metadata={
                "mode": "identity_ar_pipeline",
                "prompt_mode": "text",
                "input_tokens": len(tokens),
                "prefill_tokens": len(tokens),
                "token_count": len(tokens),
                "generated_tokens": len(tokens),
                "stop_reason": "identity",
                "streaming": False,
                "kv_cache": False,
                "kv_cache_type": None,
                "kv_cache_backend": None,
                "kv_cache_source": None,
                "runtime_kv_manager": False,
            },
        )

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        text = " ".join(prompt.strip().split())
        return text or "high quality video"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.split()

    @staticmethod
    def _stable_token_id(token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)


class TransformersARPipeline(ARPipeline):
    """Local Transformers CausalLM backend with explicit prefill/decode split.

    The stepwise loop must be bit-identical to
    ``model.generate(do_sample=False, pad_token_id=...)`` with an optional
    internal stage token budget.
    To guarantee that, ``prefill`` builds the logits processors and stopping
    criteria through transformers' own ``_prepare_generation_config`` /
    ``_get_logits_processor`` / ``_get_stopping_criteria`` -- so checkpoint
    generation_config settings such as ``repetition_penalty``,
    ``no_repeat_ngram_size``, ``min_new_tokens``, and multi-token
    ``eos_token_id`` lists behave exactly as they do under ``generate()``
    instead of being re-implemented (and drifting) here.
    """

    def __init__(
        self,
        model: str,
        *,
        device: str = "cuda",
        dtype: Any = None,
        local_files_only: bool = True,
        token_budget: int | None = None,
        prompt_mode: str = AR_PROMPT_MODE_TEXT,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.dtype = dtype or torch.bfloat16
        if prompt_mode not in SUPPORTED_AR_PROMPT_MODES:
            raise ValueError(f"Unsupported AR prompt_mode={prompt_mode!r}.")
        self.token_budget = token_budget
        self.prompt_mode = prompt_mode
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            dtype=self.dtype,
            local_files_only=local_files_only,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def generate(self, request: OmniRequest) -> ARTextOutput:
        """Run one request to completion via the stepwise primitives.

        Kept as the required ``ARPipeline`` entry point for direct use;
        ARExecutor calls ``prefill``/``decode_step``/``finalize`` itself so it
        can advance one step per ``forward()`` call.
        """
        state = self.prefill(request)
        while not state.finished:
            state = self.decode_step(state)
        return self.finalize(state)

    def prefill(self, request: OmniRequest) -> ARDecodeState:
        """Tokenize, run the full-prompt forward, and emit the first token."""
        import torch
        from transformers.generation import LogitsProcessorList, StoppingCriteriaList

        inputs = self._tokenize_prompt(request.prompt)
        input_ids = inputs.input_ids
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        input_length = int(input_ids.shape[-1])

        # Build the exact processor/criteria stack model.generate() would use
        # for the equivalent call. Pure AR text uses the model/context stop
        # policy; bridge stages may set a small internal token budget.
        generation_kwargs = {
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.token_budget is not None:
            generation_kwargs["max_new_tokens"] = self.token_budget
        generation_config, _ = self.model._prepare_generation_config(None, **generation_kwargs)
        context_window = self._resolve_context_window(input_length)
        if self.token_budget is not None:
            generation_config.max_length = input_length + self.token_budget
        else:
            generation_config.max_new_tokens = None
            generation_config.max_length = context_window
        self.model._prepare_special_tokens(
            generation_config,
            kwargs_has_attention_mask=True,
            device=input_ids.device,
            batch_size=int(input_ids.shape[0]),
        )
        logits_processor = self.model._get_logits_processor(
            generation_config,
            input_ids_seq_length=input_length,
            encoder_input_ids=None,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
            device=input_ids.device,
        )
        stopping_criteria = self.model._get_stopping_criteria(generation_config, StoppingCriteriaList())

        state = ARDecodeState(
            request_id=request.request_id,
            prompt=request.prompt,
            input_length=input_length,
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            token_budget=self.token_budget,
            context_window=context_window,
        )
        return self._step(state)

    def decode_step(self, state: ARDecodeState) -> ARDecodeState:
        """Advance one token using the KV cache built by prefill.

        The finished guard is unreachable via ARExecutor (it never STEPs a
        finished state) and via generate(); it is kept as an idempotency
        guard for direct callers of the public stepwise API.
        """
        if state.finished:
            return state
        return self._step(state)

    def finalize(self, state: ARDecodeState) -> ARTextOutput:
        """Decode the generated ids into the stage output.

        Matches the old ``model.generate()``-based path exactly: ``token_ids``
        keep a trailing stop token if one was generated (generate() includes
        it in the returned sequence); only ``text`` drops it, via
        ``skip_special_tokens``.
        """
        generated_ids = state.input_ids[0, state.input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not text:
            text = state.prompt.strip()
        token_ids = [int(item) for item in generated_ids.detach().cpu().tolist()]
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        cache_type = type(state.cache).__name__ if state.cache is not None else None
        return ARTextOutput(
            request_id=state.request_id,
            text=text,
            tokens=tokens,
            token_ids=token_ids,
            metadata={
                "mode": "transformers_causal_lm",
                "model": self.model_path,
                "prompt_mode": self.prompt_mode,
                "input_tokens": state.input_length,
                "prefill_tokens": state.input_length,
                "token_count": len(token_ids),
                "generated_tokens": len(token_ids),
                "stop_reason": self._stop_reason(len(token_ids), state),
                "streaming": False,
                "kv_cache": state.cache is not None,
                "kv_cache_type": cache_type,
                "kv_cache_backend": cache_type,
                "kv_cache_source": "transformers_past_key_values" if state.cache is not None else None,
                "runtime_kv_manager": False,
            },
        )

    def stream_events(self, state: ARDecodeState, emitted_token_count: int) -> list[ARStepOutput]:
        generated_ids = state.input_ids[0, state.input_length:]
        total = int(generated_ids.shape[-1])
        if emitted_token_count >= total:
            return []
        events: list[ARStepOutput] = []
        for token_offset in range(emitted_token_count, total):
            token_id = int(generated_ids[token_offset].detach().cpu().item())
            prev_ids = generated_ids[:token_offset]
            cur_ids = generated_ids[: token_offset + 1]
            prev_text = self.tokenizer.decode(prev_ids, skip_special_tokens=True) if token_offset > 0 else ""
            cur_text = self.tokenizer.decode(cur_ids, skip_special_tokens=True)
            text_delta = cur_text[len(prev_text):]
            events.append(
                ARStepOutput(
                    request_id=state.request_id,
                    token_index=token_offset,
                    token_id=token_id,
                    token=self.tokenizer.convert_ids_to_tokens([token_id])[0],
                    text_delta=text_delta,
                    finished=state.finished and token_offset == total - 1,
                    metadata={
                        "mode": "transformers_causal_lm",
                        "model": self.model_path,
                        "prompt_mode": self.prompt_mode,
                    },
                )
            )
        return events

    def _step(self, state: ARDecodeState) -> ARDecodeState:
        """One model forward + logits processors + argmax + stop check.

        Mirrors one iteration of transformers' greedy ``_sample`` loop: the
        last-position logits are copied to float32 before the processors run
        (so lower-precision checkpoints tie-break identically), the processors
        see the full sequence so far, and the stopping criteria run after the
        token is appended -- which is why generate() includes a generated EOS
        in its output.
        """
        import torch

        step_input_ids = state.input_ids if state.cache is None else state.input_ids[:, -1:]
        # logits_to_keep=1 matches generate(): the lm_head runs only on the
        # last position. Beyond skipping O(prompt_len x vocab) prefill work,
        # the matmul shape matters for bit-exactness -- a [1, L, H] vs
        # [1, 1, H] head matmul can use different kernels/accumulation order
        # on GPU and flip argmax ties in low precision. Gated exactly like
        # generate() gates it: remote-code models may not accept the kwarg.
        step_kwargs = {}
        supports = getattr(self.model, "_supports_logits_to_keep", None)
        if callable(supports) and supports():
            step_kwargs["logits_to_keep"] = 1
        with torch.no_grad():
            out = self.model(
                input_ids=step_input_ids,
                attention_mask=state.attention_mask,
                past_key_values=state.cache,
                use_cache=True,
                **step_kwargs,
            )
        state.cache = out.past_key_values
        logits = out.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=state.input_ids.device)
        scores = state.logits_processor(state.input_ids, logits)
        next_token = torch.argmax(scores, dim=-1)
        state.input_ids = torch.cat([state.input_ids, next_token[:, None]], dim=-1)
        state.attention_mask = torch.cat(
            [state.attention_mask, state.attention_mask.new_ones((state.attention_mask.shape[0], 1))],
            dim=-1,
        )
        # Second arg mirrors _sample's `scores` accumulator, which is None
        # unless output_scores is requested; standard criteria ignore it.
        is_done = state.stopping_criteria(state.input_ids, None)
        state.finished = bool(torch.as_tensor(is_done).all())
        return state

    def _tokenize_prompt(self, prompt: str):
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                self._build_messages(prompt),
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)
        return self.tokenizer(self._build_prompt(prompt), return_tensors="pt").to(self.device)

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        if self.prompt_mode == AR_PROMPT_MODE_I2V_BRIDGE:
            return [
                {
                    "role": "system",
                    "content": (
                        "You produce only short supplemental motion/camera hints for Wan image-to-video generation. "
                        "Do not mention subject, breed, scene, lighting, style, quality, or composition. "
                        "Return 3-6 comma-separated short phrases, max 25 words total. "
                        "Do not output JSON, markdown, bullets, or explanations."
                    ),
                },
                {"role": "user", "content": prompt.strip()},
            ]
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.strip()},
        ]

    @staticmethod
    def _stop_reason(generated_count: int, state: ARDecodeState) -> str:
        if state.token_budget is not None and generated_count >= state.token_budget:
            return "token_budget"
        if state.context_window is not None and int(state.input_ids.shape[-1]) >= state.context_window:
            return "context_limit"
        return "eos"

    def _resolve_context_window(self, input_length: int) -> int:
        candidates: list[int] = []
        for value in (
            getattr(self.tokenizer, "model_max_length", None),
            getattr(getattr(self.model, "config", None), "max_position_embeddings", None),
        ):
            try:
                candidate = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if input_length < candidate < 1_000_000:
                candidates.append(candidate)
        if candidates:
            return min(candidates)
        return max(input_length + 4096, 4096)

    def _build_prompt(self, prompt: str) -> str:
        text = prompt.strip()
        if self.prompt_mode == AR_PROMPT_MODE_I2V_BRIDGE:
            return (
                "Produce only short supplemental motion/camera hints for the following Wan image-to-video request. "
                "Do not mention subject, breed, scene, lighting, style, quality, or composition. "
                "Return 3-6 comma-separated short phrases, max 25 words total. "
                "Do not output JSON, markdown, bullets, or explanations.\n\n"
                f"Request: {text}\nPrompt:"
            )
        return text
