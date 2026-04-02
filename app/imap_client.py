import ssl
import imaplib
from datetime import datetime, timedelta, timezone

from .config import IMAP_HOST, IMAP_PORT, MAIL_LOGIN, MAIL_PASSWORD, MAILBOX, TAIL_UIDS_TO_CHECK
from .email_parser import parse_imap_message


def fetch_recent_emails(minutes=30, tail_uids=None):
    tail_uids = TAIL_UIDS_TO_CHECK if tail_uids is None else tail_uids
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=minutes)

    mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT, ssl_context=ssl.create_default_context())
    mail.login(MAIL_LOGIN, MAIL_PASSWORD)
    mail.select(MAILBOX)

    typ, data = mail.uid('search', None, 'ALL')
    uids = data[0].split() if (typ == 'OK' and data and data[0]) else []
    if not uids:
        mail.logout()
        return []

    selected = uids[-tail_uids:] if tail_uids else uids
    result = []
    for uid in selected:
        typ, msg_data = mail.uid('fetch', uid, '(RFC822)')
        if typ != 'OK' or not msg_data or msg_data[0] is None:
            continue
        raw = msg_data[0][1]
        parsed = parse_imap_message(raw)
        dt = parsed['date']
        if dt is not None and dt < cutoff:
            continue
        result.append({
            'uid': uid.decode(),
            'from': parsed['from'],
            'date': dt.isoformat() if dt else None,
            'subject': parsed['subject'],
            'text_plain': parsed['text_plain'],
            'text_html': parsed['text_html'],
        })

    mail.logout()
    return result
