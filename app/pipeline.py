import time
from datetime import datetime

from .config import WINDOW_MINUTES, validate_required, DELIVERY_MODE, OUTBOX_DIR
from .telegram_client import send_message
from .outbox_client import save_to_outbox
from .imap_client import fetch_recent_emails
from .preprocess import preprocess_for_model
from .classifier import predict_importance
from .summarizer import llm_summarize, fallback_summarize
from .state import load_state, save_state, mark_seen


def _build_summary_blocks(important):
    blocks = []
    for i, m in enumerate(important, 1):
        body = (m.get('text_plain') or m.get('text_html') or '').strip().replace('\n', ' ')
        body = ' '.join(body.split())[:1200]
        blocks.append(
            f"[{i}] From: {m['from']}\n"
            f"Date: {m['date']}\n"
            f"Subject: {m['subject']}\n"
            f"Body: {body}\n"
            f"Score: {m['proba']:.4f}\n"
        )
    return '\n\n'.join(blocks)


def run_once():
    validate_required()
    state = load_state()
    seen = set(state.get('seen_uids', []))

    mails = fetch_recent_emails(minutes=WINDOW_MINUTES)
    mails = [m for m in mails if m['uid'] not in seen]

    if not mails:
        return {'status': 'ok', 'msg': 'Нет новых писем в окне или всё уже обработано.'}

    important = []
    for m in mails:
        text_model = preprocess_for_model(m['subject'], m['text_plain'], m['text_html'])
        label, proba = predict_importance(text_model)
        m['label'] = label
        m['proba'] = proba
        if label == 1:
            important.append(m)

    mark_seen(state, [m['uid'] for m in mails])
    save_state(state)

    if not important:
        return {'status': 'ok', 'msg': f'Новых писем: {len(mails)}. Важных нет.'}

    emails_block = _build_summary_blocks(important)
    try:
        summary = llm_summarize(emails_block)
    except Exception:
        summary = fallback_summarize(important)

    tg_text = f"⚠️ Важные письма за {WINDOW_MINUTES} мин: {len(important)}\n\n{summary}"
    if DELIVERY_MODE == "telegram":
        send_message(tg_text)
    else:
        save_to_outbox(
            text=tg_text,
            payload={
                "important_uids": [m["uid"] for m in important],
                "important_count": len(important),
            },
            outbox_dir=OUTBOX_DIR,
        )

    return {
        'status': 'sent',
        'new_total': len(mails),
        'important': len(important),
        'important_uids': [m['uid'] for m in important],
    }


def run_forever():
    while True:
        try:
            print(datetime.now().isoformat(), run_once())
        except Exception as e:
            print(datetime.now().isoformat(), 'ERROR:', repr(e))
        time.sleep(WINDOW_MINUTES * 60)
