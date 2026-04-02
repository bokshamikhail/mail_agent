import json
from .config import STATE_PATH


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding='utf-8'))
    return {'seen_uids': []}


def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')


def mark_seen(state: dict, uids):
    seen = set(state.get('seen_uids', []))
    for uid in uids:
        seen.add(str(uid))
    state['seen_uids'] = list(seen)[-5000:]
