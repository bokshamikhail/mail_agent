import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

_tokenizer = None
_model = None


def _apply_chat_template(tokenizer, messages):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _load_llm():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    models_dir = os.getenv("MODELS_DIR", "/home/miboksha/VKR_Boksha_Mikhail/models")
    model_path = os.path.join(models_dir, "Qwen3-8B")

    _tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,   # КРИТИЧНО для Qwen3
        use_fast=False,           # чтобы не падал tokenizer.json/ModelWrapper
    )

    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,   # КРИТИЧНО для Qwen3
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    _model.eval()
    return _tokenizer, _model


@torch.no_grad()
def llm_summarize(text: str, max_new_tokens: int = 500) -> str:
    tok, model = _load_llm()

    messages = [
        {"role": "system", "content": "Ты полезный ассистент. Отвечай по-русски кратко и по делу."},
        {"role": "user", "content": text},
    ]

    prompt = _apply_chat_template(tok, messages)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        min_new_tokens=1,
    )

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def fallback_summarize(items):
    # простой fallback: список важных писем
    lines = []
    for i, m in enumerate(items, 1):
        subj = (m.get("subject") or "").strip()
        frm = (m.get("from") or "").strip()
        score = m.get("proba", m.get("score"))
        if score is not None:
            try:
                lines.append(f"• #{i} {subj} (от {frm}) | score={float(score):.3f}")
            except Exception:
                lines.append(f"• #{i} {subj} (от {frm}) | score={score}")
        else:
            lines.append(f"• #{i} {subj} (от {frm})")
    return "\n".join(lines).strip()