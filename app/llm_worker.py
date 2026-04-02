import json, os, sys
from pathlib import Path

def build_prompt(emails):
    n = len(emails)

    blocks = []
    for i, m in enumerate(emails, 1):
        frm = (m.get("from", "") or "").strip()
        subj = (m.get("subject", "") or "").strip()
        dt = (m.get("date", "") or "").strip()

        body = (m.get("text", "") or "").strip()
        body = " ".join(body.split())
        body = body[:600]  # не даём модели залипать на одном длинном письме

        blocks.append(
            f"EMAIL {i}\n"
            f"FROM: {frm}\n"
            f"SUBJECT: {subj}\n"
            f"DATE: {dt}\n"
            f"BODY: {body}\n"
        )

    emails_block = "\n\n".join(blocks)

    return f"""Ты — корпоративный ассистент. Суммаризируй КАЖДОЕ письмо ОТДЕЛЬНО.
            ЖЁСТКИЕ ПРАВИЛА:
            1) Выведи МАКСИМУМ {2*n} строк — не больше двух строке на каждое письмо EMAIL 1..EMAIL {n}.
            2) Каждая строка строго в формате:
               [EMAIL <номер>] <краткая суть (до 12 слов), с упоминанием курса, если он есть в письме> | действие: <что сделать/когда/кому или "нет">
            3) Никаких вступлений/заголовков/общей сводки. Не объединяй письма. Не пропускай письма.
            4) Если письма похожи, всё равно дай строку для каждого (можно пометить "дубликат").
            ПИСЬМА:
            {emails_block}
            """

def main():
    if len(sys.argv) < 3:
        print("Usage: llm_worker.py input.json output.txt", file=sys.stderr)
        return 2

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    data = json.loads(in_path.read_text(encoding="utf-8"))
    emails = data.get("emails", [])
    prompt = build_prompt(emails)

    model_dir = os.getenv("QWEN_DIR", "/home/miboksha/VKR_Boksha_Mikhail/models/Qwen3-8B")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    )

    messages = [
        {"role": "system", "content": "Ты корпоративный ассистент. Пиши по-русски кратко и по делу."},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    n = len(emails)
    max_new_tokens = min(2048, 120 * n + 200)
    inputs = tok(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    out_path.write_text(text.strip(), encoding="utf-8")
    print("OK", out_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
