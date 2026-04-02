import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
from datetime import timezone


def decode_mime(s):
    if not s:
        return ''
    parts = decode_header(s)
    out = []
    for text, enc in parts:
        if isinstance(text, bytes):
            enc = (enc or 'utf-8').lower()
            if enc in ('unknown-8bit', 'unknown', 'x-unknown'):
                enc = 'utf-8'
            try:
                out.append(text.decode(enc, errors='replace'))
            except LookupError:
                out.append(text.decode('utf-8', errors='replace'))
        else:
            out.append(text)
    return ''.join(out)


def extract_text(msg):
    text_plain, text_html = None, None
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get('Content-Disposition', ''))
            if 'attachment' in disp.lower():
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            charset = part.get_content_charset() or 'utf-8'
            decoded = payload.decode(charset, errors='replace')
            if ctype == 'text/plain' and text_plain is None:
                text_plain = decoded
            elif ctype == 'text/html' and text_html is None:
                text_html = decoded
    else:
        ctype = msg.get_content_type()
        payload = msg.get_payload(decode=True) or b''
        charset = msg.get_content_charset() or 'utf-8'
        decoded = payload.decode(charset, errors='replace')
        if ctype == 'text/plain':
            text_plain = decoded
        elif ctype == 'text/html':
            text_html = decoded
    return (text_plain or '').strip(), (text_html or '').strip()


def parse_imap_message(raw_bytes):
    msg = email.message_from_bytes(raw_bytes)
    subject = decode_mime(msg.get('Subject'))
    from_ = decode_mime(msg.get('From'))
    date_raw = msg.get('Date')
    dt = None
    if date_raw:
        try:
            dt = parsedate_to_datetime(date_raw)
            if dt and dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            dt = None
    text_plain, text_html = extract_text(msg)
    return {
        'subject': subject,
        'from': from_,
        'date': dt,
        'text_plain': text_plain,
        'text_html': text_html,
    }
