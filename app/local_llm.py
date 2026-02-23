"""
local_llm.py – optional local quantized model (GPU only).

Only imported when a GPU is available and LOCAL_MODEL env var is set.
Requires: transformers, accelerate, bitsandbytes
"""
import os
import torch
from app.device import has_gpu

_model = None
_tokenizer = None
_loaded = False


def _ensure_loaded():
    global _model, _tokenizer, _loaded
    if _loaded:
        return
    if not has_gpu():
        raise RuntimeError("local_llm requires a CUDA GPU")

    model_name = os.environ.get("LOCAL_MODEL", "")
    if not model_name:
        raise RuntimeError("Set LOCAL_MODEL env var to the HuggingFace model id/path")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"⏳ Loading local model: {model_name} (8-bit quantised) …")
    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    _loaded = True
    print(f"✅ Local model loaded on {_model.device}")


def generate_local(prompt: str, max_new_tokens: int = 256) -> str:
    """Generate text from a local quantised model."""
    _ensure_loaded()
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )
    return _tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
