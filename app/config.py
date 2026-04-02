import os
from pathlib import Path
def _load_env_file(path: str = None):
    import os
    from pathlib import Path

    if path is None:
        base = Path(__file__).resolve().parents[1]  # .../VKR_Boksha_Mikhail
        path = str(base / ".env")

    p = Path(path)
    if not p.exists():
        return

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)

_load_env_file()

IMAP_HOST = os.getenv('IMAP_HOST', 'imap.yandex.ru')
IMAP_PORT = int(os.getenv('IMAP_PORT', '993'))
MAIL_LOGIN = os.getenv('MAIL_LOGIN')
MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
MAILBOX = os.getenv('MAILBOX', 'INBOX')

TG_BOT_TOKEN = os.getenv('TG_BOT_TOKEN')
TG_CHAT_ID = os.getenv('TG_CHAT_ID')

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv('MODELS_DIR', str(BASE_DIR / 'models')))

FASTTEXT_PATH = MODELS_DIR / 'fasttext_unsup.bin'
CATBOOST_PATH = MODELS_DIR / 'catboost_final.cbm'
QWEN_PATH = MODELS_DIR / 'Qwen3-8B'

USE_LOCAL_LLM = os.getenv('USE_LOCAL_LLM', '1') == '1'
IMPORTANT_THRESHOLD = float(os.getenv('IMPORTANT_THRESHOLD', '0.315'))
WINDOW_MINUTES = int(os.getenv('WINDOW_MINUTES', '30'))
TAIL_UIDS_TO_CHECK = int(os.getenv('TAIL_UIDS_TO_CHECK', '250'))
STATE_PATH = Path(os.getenv('STATE_PATH', str(BASE_DIR / 'state.json')))
RUN_MODE = os.getenv('RUN_MODE', 'once')
DELIVERY_MODE = os.getenv('DELIVERY_MODE', 'telegram')  # telegram | outbox
OUTBOX_DIR = Path(os.getenv('OUTBOX_DIR', str(BASE_DIR / 'outbox')))


def validate_required():
    missing = []
    for key, val in [('MAIL_LOGIN', MAIL_LOGIN), ('MAIL_PASSWORD', MAIL_PASSWORD)]:
        if not val:
            missing.append(key)

    if DELIVERY_MODE == 'telegram':
        for key, val in [('TG_BOT_TOKEN', TG_BOT_TOKEN), ('TG_CHAT_ID', TG_CHAT_ID)]:
            if not val:
                missing.append(key)

    if missing:
        raise RuntimeError(f"Не заданы переменные окружения: {', '.join(missing)}")