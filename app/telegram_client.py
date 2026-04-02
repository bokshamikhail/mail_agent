import requests
from .config import TG_BOT_TOKEN, TG_CHAT_ID


def send_message(text: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        raise RuntimeError('Не заданы TG_BOT_TOKEN или TG_CHAT_ID в .env')
    url = f'https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage'
    resp = requests.post(url, json={
        'chat_id': TG_CHAT_ID,
        'text': text,
        'disable_web_page_preview': True,
    }, timeout=20)
    resp.raise_for_status()
    return resp.json()
