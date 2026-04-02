import re
import spacy

_NLP_RU = None


def clean_text(x: str) -> str:
    if not isinstance(x, str):
        return ''
    x = re.sub(r'\s+', ' ', x)
    return x.strip()


def build_text(subject: str, text_plain: str, text_html: str) -> str:
    subj = clean_text(subject)
    body = text_plain if (isinstance(text_plain, str) and text_plain.strip()) else (text_html or '')
    body = clean_text(body)
    return f'SUBJECT: {subj}\nBODY: {body}'.strip()


def clear_text(text: str) -> str:
    text = (text or '').lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _get_nlp():
    global _NLP_RU
    if _NLP_RU is None:
        _NLP_RU = spacy.load('ru_core_news_sm', disable=['parser', 'ner'])
    return _NLP_RU


def lemmatize_ru(text: str) -> str:
    nlp = _get_nlp()
    doc = nlp(text)
    return ' '.join(t.lemma_ for t in doc).strip()


def preprocess_for_model(subject: str, text_plain: str, text_html: str) -> str:
    text = build_text(subject, text_plain, text_html)
    text = clear_text(text)
    text = lemmatize_ru(text)
    return text
