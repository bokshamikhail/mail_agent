from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Импорты из проекта
from app.preprocess import preprocess_for_model
from app.classifier import predict_importance
from app.summarizer import llm_summarize, fallback_summarize
import traceback
from dotenv import load_dotenv
load_dotenv()

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_job_id_from_name(p: Path) -> str:
    # job_<id>.json -> <id>
    name = p.stem
    if name.startswith("job_"):
        return name[len("job_") :]
    return name


def _build_summary_blocks(important):
    blocks = []
    for i, m in enumerate(important, 1):
        uid = m.get("uid", "")
        subj = (m.get("subject") or "").strip()
        frm = (m.get("from") or "").strip()
        dt = (m.get("date") or "").strip()

        body = (m.get("text_plain") or m.get("text_html") or "").strip()
        body = " ".join(body.split())
        body = body[:600]

        blocks.append(
            f"EMAIL {i} (uid={uid})\n"
            f"FROM: {frm}\n"
            f"DATE: {dt}\n"
            f"SUBJECT: {subj}\n"
            f"BODY: {body}\n"
        )
    return "\n\n".join(blocks)


def process_job(job_path: Path, out_dir: Path) -> Path:
    job: Dict[str, Any] = json.loads(job_path.read_text(encoding="utf-8"))
    job_id = str(job.get("job_id") or _safe_job_id_from_name(job_path))
    window_minutes = int(job.get("window_minutes") or 30)
    emails: List[Dict[str, Any]] = list(job.get("emails") or [])
    llm_error = ""
    processed_at = _utc_now_iso()

    important: List[Dict[str, Any]] = []
    processed_all: List[Dict[str, Any]] = []

    for m in emails:
        subject = m.get("subject")
        text_plain = m.get("text_plain")
        text_html = m.get("text_html")
        text_model = preprocess_for_model(subject, text_plain, text_html)
        label, proba = predict_importance(text_model)
        m2 = dict(m)
        m2["label"] = int(label)
        m2["proba"] = float(proba)
        processed_all.append(m2)
        if label == 1:
            important.append(m2)

    if important:
        emails_block = _build_summary_blocks(important)
        try:
            summary = llm_summarize(emails_block)
        except Exception as e:
            tb = traceback.format_exc()
            print("=== LLM ERROR TRACEBACK ===")
            print(tb)
            print("===========================")

            summary_fb = fallback_summarize(important)
            summary = f"(LLM ошибка: {type(e).__name__}: {e})\n" + summary_fb
            llm_error = tb
            
        tg_text = f"⚠️ Важные письма за {window_minutes} мин: {len(important)}\n\n{summary}"
    else:
        summary = ""
        tg_text = f"Новых важных писем за {window_minutes} мин нет."

    result = {
        "job_id": job_id,
        "created_at": job.get("created_at"),
        "processed_at": processed_at,
        "window_minutes": window_minutes,
        "new_total": len(emails),
        "important_count": len(important),
        "important_uids": [m.get("uid") for m in important],
        "emails": processed_all,
        "summary": summary,
        "tg_text": tg_text,
        "llm_error": llm_error if important else "",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"result_{job_id}.json"
    tmp_path = out_dir / f".tmp_result_{job_id}.json"
    tmp_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-dir", default=None, help="Папка jobs (по умолчанию: <repo>/jobs)")
    ap.add_argument("--once", action="store_true", help="Сделать один проход и выйти")
    ap.add_argument("--sleep", type=int, default=1800, help="Пауза между проходами в режиме forever")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    jobs_dir = Path(args.jobs_dir).expanduser() if args.jobs_dir else (base / "jobs")
    in_dir = jobs_dir / "in"
    out_dir = jobs_dir / "out"
    done_dir = jobs_dir / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sent").mkdir(parents=True, exist_ok=True)

    def _pass_once() -> None:
        in_dir.mkdir(parents=True, exist_ok=True)
        job_files = sorted(in_dir.glob("job_*.json"), key=lambda p: p.stat().st_mtime)
        if not job_files:
            print(_utc_now_iso(), "No jobs")
            return
        for jp in job_files:
            try:
                print(_utc_now_iso(), "Processing", jp.name)
                out_path = process_job(jp, out_dir)
                # перемещаем job в done
                dest = done_dir / jp.name
                shutil.move(str(jp), str(dest))
                print(_utc_now_iso(), "OK ->", out_path.name)
            except Exception as e:
                # чтобы не зациклиться на битом файле — перекинем в done с суффиксом .error
                err_dest = done_dir / (jp.name + ".error")
                try:
                    shutil.move(str(jp), str(err_dest))
                except Exception:
                    pass
                print(_utc_now_iso(), "ERROR", jp.name, repr(e))

    if args.once:
        _pass_once()
        return 0

    while True:
        _pass_once()
        time.sleep(int(args.sleep))


if __name__ == "__main__":
    raise SystemExit(main())
