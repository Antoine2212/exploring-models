# qwen_cache.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import copy

import torch

from gemma_explore.qwen_core import (
    DEFAULT_MODEL,
    FrequencyScoreOutputs,
    PromptEncoding,
    QwenBundle,
    ScoreOutputs,
    collect_frequency_scores,
    collect_scores,
    compute_attention_internals_for_layer,
    encode_prompt,
    extract_full_attention_matrices,
    get_decoder_layers,
    prompt_text_hash,
    run_model,
    to_cpu,
    token_ids_hash,
)


SCHEMA_VERSION = 3


@dataclass
class CacheHandle:
    """Notebook-friendly wrapper around the cache payload."""

    payload: dict[str, Any]
    path: Path | None = None

    @property
    def model_id(self) -> str:
        return str(self.payload.get("model_id", ""))

    @property
    def prompts(self) -> list[dict[str, Any]]:
        return self.payload.setdefault("prompts", [])

    @property
    def derived(self) -> dict[str, Any]:
        return self.payload.setdefault("derived", {"scores": {}, "frequency_scores": {}})

    @property
    def meta(self) -> dict[str, Any]:
        return self.payload.setdefault("meta", {})

    def get_prompt(self, prompt_id: int) -> dict[str, Any]:
        return get_prompt_record(self, prompt_id)

    def get_layer(self, prompt_id: int, layer_idx: int) -> dict[str, Any]:
        return get_layer_record(self, prompt_id, layer_idx)


# ---------------------------------------------------------------------
# Schema / upgrade
# ---------------------------------------------------------------------


def _empty_payload(model_id: str = DEFAULT_MODEL, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "config": dict(config or {}),
        "meta": {},
        "prompts": [],
        "derived": {
            "scores": {},
            "frequency_scores": {},
        },
    }


def _cpuify(obj: Any) -> Any:
    """Recursively move tensors to CPU for serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _cpuify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpuify(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpuify(v) for v in obj)
    return obj


def _ensure_prompt_metadata(prompt_record: dict[str, Any]) -> None:
    prompt_record.setdefault("prompt_id", len(prompt_record))
    text = str(prompt_record.get("text", ""))
    token_ids = prompt_record.get("input_ids")
    if isinstance(token_ids, torch.Tensor) and token_ids.ndim == 2 and token_ids.shape[0] == 1:
        prompt_ids = token_ids[0].tolist()
    elif isinstance(token_ids, torch.Tensor) and token_ids.ndim == 1:
        prompt_ids = token_ids.tolist()
    elif isinstance(token_ids, (list, tuple)):
        prompt_ids = list(token_ids[0] if token_ids and isinstance(token_ids[0], (list, tuple)) else token_ids)
    else:
        prompt_ids = []

    prompt_record.setdefault("text_hash", prompt_text_hash(text))
    prompt_record.setdefault("token_ids_hash", token_ids_hash(prompt_ids) if prompt_ids else None)
    prompt_record.setdefault("seq_len", len(prompt_ids))
    prompt_record.setdefault("tokens", prompt_record.get("tokens", []))
    prompt_record.setdefault("blocks", [])


def upgrade_cache_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade older cache layouts to the current schema."""
    upgraded = copy.deepcopy(dict(payload))

    if "schema_version" in upgraded and "derived" in upgraded:
        upgraded.setdefault("meta", {})
        upgraded.setdefault("prompts", [])
        stored_version = int(upgraded["schema_version"])
        upgraded["schema_version"] = SCHEMA_VERSION
        upgraded["derived"].setdefault("scores", {})
        upgraded["derived"].setdefault("frequency_scores", {})
        if stored_version < 3:
            # Scores computed before v3 used input-level permutation; discard them
            # so they are recomputed with the new hidden-state permutation method.
            upgraded["derived"]["scores"] = {}
            upgraded["derived"]["frequency_scores"] = {}
        for i, rec in enumerate(upgraded["prompts"]):
            rec.setdefault("prompt_id", i)
            if "attentions" in rec and "attentions_hf" not in rec:
                rec["attentions_hf"] = rec.pop("attentions")
            _ensure_prompt_metadata(rec)
        return upgraded

    normalized = _empty_payload(
        model_id=str(upgraded.get("model_id", DEFAULT_MODEL)),
        config=upgraded.get("config", {}),
    )
    normalized["meta"] = dict(upgraded.get("meta", {}))
    normalized["prompts"] = list(upgraded.get("prompts", []))

    for i, rec in enumerate(normalized["prompts"]):
        rec.setdefault("prompt_id", i)
        if "attentions" in rec and "attentions_hf" not in rec:
            rec["attentions_hf"] = rec.pop("attentions")
        _ensure_prompt_metadata(rec)

    old_scores = upgraded.get("scores")
    old_freq = upgraded.get("frequency_scores")
    if isinstance(old_scores, dict):
        normalized["derived"]["scores"] = old_scores
    if isinstance(old_freq, dict):
        normalized["derived"]["frequency_scores"] = old_freq

    return normalized


def load_cache(path: str | Path, map_location: str | torch.device = "cpu") -> CacheHandle:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    return CacheHandle(payload=upgrade_cache_payload(payload), path=Path(path))


def save_cache(cache: CacheHandle | Mapping[str, Any], path: str | Path | None = None) -> Path:
    payload = cache.payload if isinstance(cache, CacheHandle) else dict(cache)
    out_path = Path(path) if path is not None else None
    if out_path is None:
        if isinstance(cache, CacheHandle) and cache.path is not None:
            out_path = cache.path
        else:
            raise ValueError("save_cache requires an explicit path when cache.path is unknown")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_cpuify(payload), out_path)
    if isinstance(cache, CacheHandle):
        cache.path = out_path
    return out_path


# ---------------------------------------------------------------------
# Prompt / layer accessors
# ---------------------------------------------------------------------


def get_prompt_record(cache: CacheHandle | Mapping[str, Any], prompt_id: int) -> dict[str, Any]:
    payload = cache.payload if isinstance(cache, CacheHandle) else dict(cache)
    prompts = payload.get("prompts", [])
    if not (0 <= prompt_id < len(prompts)):
        raise IndexError(f"prompt_id={prompt_id} out of range for {len(prompts)} prompts")
    return prompts[prompt_id]


def get_layer_record(cache: CacheHandle | Mapping[str, Any], prompt_id: int, layer_idx: int) -> dict[str, Any]:
    prompt = get_prompt_record(cache, prompt_id)
    layers = prompt.get("layers", [])
    if not (0 <= layer_idx < len(layers)):
        raise IndexError(f"layer_idx={layer_idx} out of range for {len(layers)} layers")
    return layers[layer_idx]


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------


def _bundle_meta(bundle: QwenBundle) -> dict[str, Any]:
    return {
        "num_layers": bundle.num_layers,
        "num_q_heads": bundle.num_heads,
        "num_kv_heads": bundle.num_kv_heads,
        "num_kv_groups": bundle.q_group_size,
        "hidden_size": bundle.hidden_size,
        "head_dim": bundle.head_dim,
        "dtype": str(bundle.dtype),
        "device": str(bundle.device),
        "attn_implementation": getattr(bundle.model.config, "_attn_implementation", None),
        "rope_theta": bundle.rope_theta,
        "num_frequencies": bundle.num_frequencies,
    }


def _score_record_from_outputs(outputs: ScoreOutputs, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "scores",
        "params": dict(params),
        "swaps": list(outputs.swaps),
        "blocks": list(outputs.blocks),
        "pos_scores": to_cpu(outputs.pos_scores),
        "sym_scores": to_cpu(outputs.sym_scores),
        "mean_scores": to_cpu(outputs.mean_scores),
        "att_last_col": to_cpu(outputs.att_last_col),
        "tokens": list(outputs.tokens),
        "prompt_token_ids": list(outputs.prompt_token_ids),
        "prompt_text": outputs.prompt_text,
        "prompt_blocks": outputs.prompt_blocks,
    }


def _frequency_record_from_outputs(outputs: FrequencyScoreOutputs, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "frequency_scores",
        "params": dict(params),
        "swaps": list(outputs.swaps),
        "blocks": list(outputs.blocks),
        "full_scores": to_cpu(outputs.full_scores),
        "frequency_scores": to_cpu(outputs.frequency_scores),
        "frequency_logit_norms": to_cpu(outputs.frequency_logit_norms),
        "frequency_wavelengths": to_cpu(outputs.frequency_wavelengths),
        "attention_matrix": to_cpu(outputs.attention_matrix),
        "tokens": list(outputs.tokens),
        "prompt_token_ids": list(outputs.prompt_token_ids),
        "prompt_text": outputs.prompt_text,
        "prompt_blocks": outputs.prompt_blocks,
    }


def _derived_key(
    prompt_record: Mapping[str, Any],
    *,
    model_id: str,
    apply_chat_template: bool,
    add_generation_prompt: bool,
    n_blocks: int,
    tau: float,
) -> str:
    return "|".join(
        [
            f"model={model_id}",
            f"prompt={prompt_record.get('prompt_id')}",
            f"text={prompt_record.get('text_hash')}",
            f"tokens={prompt_record.get('token_ids_hash')}",
            f"seq={prompt_record.get('seq_len')}",
            f"chat={int(apply_chat_template)}",
            f"gen={int(add_generation_prompt)}",
            f"blocks={int(n_blocks)}",
            f"tau={float(tau):.8g}",
        ]
    )


def _prompt_record_from_encoding(prompt_id: int, encoding: PromptEncoding) -> dict[str, Any]:
    input_ids_cpu = encoding.input_ids.detach().cpu()
    attention_mask_cpu = encoding.attention_mask.detach().cpu()
    token_ids = input_ids_cpu[0].tolist()
    return {
        "prompt_id": prompt_id,
        "messages": encoding.messages,
        "text": encoding.text,
        "text_hash": prompt_text_hash(encoding.text),
        "input_ids": input_ids_cpu,
        "attention_mask": attention_mask_cpu,
        "token_ids_hash": token_ids_hash(token_ids),
        "seq_len": len(token_ids),
        "tokens": list(encoding.tokens),
        "blocks": list(encoding.blocks),
        "attentions_hf": [],
        "layers": [],
    }


@torch.inference_mode()
def build_activation_cache(
    bundle: QwenBundle,
    prompts: Sequence[str | list[dict[str, str]]],
    apply_chat_template: bool = True,
    add_generation_prompt: bool = True,
    include_hf_attentions: bool = True,
    verbose: bool = False,
) -> CacheHandle:
    """Build a full activation cache from prompts.

    Stores raw prompt data, HF attentions, and rich per-layer reconstructed internals.
    """
    payload = _empty_payload(model_id=bundle.model_id, config=bundle.model.config.to_dict())
    payload["meta"] = _bundle_meta(bundle)

    layers = get_decoder_layers(bundle)

    for prompt_id, prompt in enumerate(prompts):
        encoding = encode_prompt(
            bundle,
            prompt,
            apply_chat_template=apply_chat_template,
            add_generation_prompt=add_generation_prompt,
        )
        out = run_model(
            bundle,
            encoding.input_ids,
            encoding.attention_mask,
            output_hidden_states=True,
            output_attentions=include_hf_attentions,
        )

        hidden_states = out.hidden_states
        prompt_record = _prompt_record_from_encoding(prompt_id, encoding)
        if include_hf_attentions:
            prompt_record["attentions_hf"] = [to_cpu(a) for a in out.attentions]

        for layer_idx, layer in enumerate(layers):
            block_in = hidden_states[layer_idx].to(layer.self_attn.q_proj.weight.device)
            block_out_hf = hidden_states[layer_idx + 1].to(layer.self_attn.q_proj.weight.device)

            internals = compute_attention_internals_for_layer(
                bundle=bundle,
                hidden_in=block_in,
                attention_mask=encoding.attention_mask,
                layer_idx=layer_idx,
            )

            hf_attn = out.attentions[layer_idx] if include_hf_attentions else None
            checks = {
                "max_abs_diff_block_output_vs_hf_hidden_state": float(
                    (internals["block_output"] - block_out_hf).abs().max().item()
                )
            }
            if hf_attn is not None:
                checks["max_abs_diff_attn_weights_vs_hf_attn"] = float(
                    (internals["attn_weights"] - hf_attn.to(internals["attn_weights"].device)).abs().max().item()
                )

            layer_record = {
                "layer_idx": layer_idx,
                "block_input": to_cpu(internals["block_input"]),
                "input_layernorm_out": to_cpu(internals["input_layernorm_out"]),
                "attn_block_out": to_cpu(internals["attn_block_out"]),
                "residual_after_attn": to_cpu(internals["residual_after_attn"]),
                "post_attn_layernorm_out": to_cpu(internals["post_attn_layernorm_out"]),
                "mlp_out": to_cpu(internals["mlp_out"]),
                "block_output": to_cpu(internals["block_output"]),
                "q_pre_rope": to_cpu(internals["q_pre_rope"]),
                "k_pre_rope": to_cpu(internals["k_pre_rope"]),
                "v_pre": to_cpu(internals["v_pre"]),
                "q_post_rope": to_cpu(internals["q_post_rope"]),
                "k_post_rope": to_cpu(internals["k_post_rope"]),
                "attn_scores": to_cpu(internals["attn_scores"]),
                "attn_weights": to_cpu(internals["attn_weights"]),
                "head_out": to_cpu(internals["head_out"]),
                "kv_head_for_q": to_cpu(internals["kv_head_for_q"]),
                "checks": checks,
            }
            prompt_record["layers"].append(layer_record)

        payload["prompts"].append(prompt_record)
        if verbose:
            print(f"[build_activation_cache] cached prompt {prompt_id} with {len(prompt_record['layers'])} layers")

    return CacheHandle(payload=payload)


# ---------------------------------------------------------------------
# Derived results management
# ---------------------------------------------------------------------


def _matching_params(record: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    stored = dict(record.get("params", {}))
    return stored == dict(params)


def ensure_scores(
    cache: CacheHandle,
    *,
    bundle: QwenBundle,
    prompt_id: int,
    n_blocks: int = 16,
    tau: float = 0.1,
    apply_chat_template: bool = True,
    add_generation_prompt: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Ensure global scores are present in cache. Reuse when params match."""
    prompt_record = get_prompt_record(cache, prompt_id)
    params = {
        "model_id": bundle.model_id,
        "prompt_id": prompt_id,
        "n_blocks": int(n_blocks),
        "tau": float(tau),
        "apply_chat_template": bool(apply_chat_template),
        "add_generation_prompt": bool(add_generation_prompt),
        "text_hash": prompt_record.get("text_hash"),
        "token_ids_hash": prompt_record.get("token_ids_hash"),
        "seq_len": prompt_record.get("seq_len"),
    }
    key = _derived_key(
        prompt_record,
        model_id=bundle.model_id,
        apply_chat_template=apply_chat_template,
        add_generation_prompt=add_generation_prompt,
        n_blocks=n_blocks,
        tau=tau,
    )

    existing = cache.derived["scores"].get(key)
    if (not force) and isinstance(existing, dict) and _matching_params(existing, params):
        return existing

    outputs = collect_scores(
        bundle=bundle,
        prompt=prompt_record["text"],
        n_blocks=n_blocks,
        tau=tau,
        apply_chat_template=False,
        add_generation_prompt=add_generation_prompt,
    )
    record = _score_record_from_outputs(outputs, params)
    cache.derived["scores"][key] = record
    return record


def ensure_frequency_scores(
    cache: CacheHandle,
    *,
    bundle: QwenBundle,
    prompt_id: int,
    n_blocks: int = 16,
    tau: float = 0.1,
    apply_chat_template: bool = True,
    add_generation_prompt: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Ensure frequency-resolved scores are present in cache."""
    prompt_record = get_prompt_record(cache, prompt_id)
    params = {
        "model_id": bundle.model_id,
        "prompt_id": prompt_id,
        "n_blocks": int(n_blocks),
        "tau": float(tau),
        "apply_chat_template": bool(apply_chat_template),
        "add_generation_prompt": bool(add_generation_prompt),
        "text_hash": prompt_record.get("text_hash"),
        "token_ids_hash": prompt_record.get("token_ids_hash"),
        "seq_len": prompt_record.get("seq_len"),
    }
    key = _derived_key(
        prompt_record,
        model_id=bundle.model_id,
        apply_chat_template=apply_chat_template,
        add_generation_prompt=add_generation_prompt,
        n_blocks=n_blocks,
        tau=tau,
    )

    existing = cache.derived["frequency_scores"].get(key)
    if (not force) and isinstance(existing, dict) and _matching_params(existing, params):
        return existing

    outputs = collect_frequency_scores(
        bundle=bundle,
        prompt=prompt_record["text"],
        n_blocks=n_blocks,
        tau=tau,
        apply_chat_template=False,
        add_generation_prompt=add_generation_prompt,
    )
    record = _frequency_record_from_outputs(outputs, params)
    cache.derived["frequency_scores"][key] = record
    return record


# ---------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------


def prompt_summary(cache: CacheHandle, prompt_id: int) -> dict[str, Any]:
    rec = get_prompt_record(cache, prompt_id)
    return {
        "prompt_id": rec.get("prompt_id"),
        "seq_len": rec.get("seq_len"),
        "text": rec.get("text"),
        "num_layers": len(rec.get("layers", [])),
        "num_tokens": len(rec.get("tokens", [])),
    }


def list_derived_keys(cache: CacheHandle, kind: str = "scores") -> list[str]:
    if kind not in cache.derived:
        raise KeyError(f"Unknown derived kind: {kind}")
    return sorted(cache.derived[kind].keys())


def get_prompt_attention_matrix(cache: CacheHandle, prompt_id: int, layer_idx: int) -> torch.Tensor:
    prompt = get_prompt_record(cache, prompt_id)
    attentions = prompt.get("attentions_hf")
    if not attentions:
        raise KeyError("HF attentions are not present in this cache")
    return attentions[layer_idx]