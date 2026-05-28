from pathlib import Path
import torch
import matplotlib.pyplot as plt
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src" 

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

print("FILE:", __file__)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("SRC_PATH exists:", SRC_PATH.exists(), SRC_PATH)
print("sys.path[0:3]:", sys.path[:3])
from gemma_explore.qwen_core import DEFAULT_MODEL, load_bundle
from gemma_explore.qwen_cache import (
    load_cache,
    save_cache,
    build_activation_cache,
    ensure_scores,
    ensure_frequency_scores,
)
from gemma_explore.qwen_viz import (
    plot_all_attention_heads,
    plot_head_block_dynamics,
    plot_pos_sym_heatmaps,
    plot_heads_scatter,
    plot_frequency_analysis,
)



CACHE_PATH = Path("data/cache/qwen_notebook_cache.pt")

PROMPT = "The punctuation, it is important! Without it, nothing can be understood."
APPLY_CHAT_TEMPLATE = False
ADD_GENERATION_PROMPT = False

bundle = load_bundle(DEFAULT_MODEL)

def get_or_create_prompt_cache(
    cache_path: Path,
    bundle,
    prompt: str,
    apply_chat_template: bool = True,
    add_generation_prompt: bool = True,
):
    if cache_path.exists():
        cache = load_cache(cache_path)
    else:
        cache = None

    if cache is not None:
        for i, rec in enumerate(cache.prompts):
            if rec.get("text") == prompt:
                print(f"Reusing cached prompt at prompt_id={i}")
                return cache, i

    new_cache = build_activation_cache(
        bundle=bundle,
        prompts=[prompt],
        apply_chat_template=apply_chat_template,
        add_generation_prompt=add_generation_prompt,
        include_hf_attentions=True,
        verbose=True,
    )

    if cache is None:
        save_cache(new_cache, cache_path)
        print("Created new cache file.")
        return new_cache, 0

    new_prompt = new_cache.prompts[0]
    new_prompt["prompt_id"] = len(cache.prompts)
    cache.prompts.append(new_prompt)
    save_cache(cache, cache_path)
    print(f"Added prompt to existing cache at prompt_id={new_prompt['prompt_id']}")
    return cache, new_prompt["prompt_id"]

if __name__ == "__main__":
    cache, prompt_id = get_or_create_prompt_cache(
        cache_path=CACHE_PATH,
        bundle=bundle,
        prompt=PROMPT,
        apply_chat_template=APPLY_CHAT_TEMPLATE,
        add_generation_prompt=ADD_GENERATION_PROMPT,
    )

    print("prompt_id =", prompt_id)
    print("num prompts in cache =", len(cache.prompts))
    print("seq_len =", cache.prompts[prompt_id]["seq_len"])