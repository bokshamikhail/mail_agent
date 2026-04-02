import json
from pathlib import Path
from datetime import datetime

def save_to_outbox(text: str, payload: dict, outbox_dir: Path):
    outbox_dir = Path(outbox_dir)
    outbox_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = outbox_dir / f"summary_{ts}.txt"
    json_path = outbox_dir / f"summary_{ts}.json"

    txt_path.write_text(text, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"txt": str(txt_path), "json": str(json_path)}