import numpy as np
import fasttext
from catboost import CatBoostClassifier

from .config import FASTTEXT_PATH, CATBOOST_PATH, IMPORTANT_THRESHOLD

_ft_model = None
_cb_model = None


def load_models():
    global _ft_model, _cb_model
    if _ft_model is None:
        if not FASTTEXT_PATH.exists():
            raise FileNotFoundError(f'Не найден fastText: {FASTTEXT_PATH}')
        _ft_model = fasttext.load_model(str(FASTTEXT_PATH))
    if _cb_model is None:
        if not CATBOOST_PATH.exists():
            raise FileNotFoundError(f'Не найден CatBoost: {CATBOOST_PATH}')
        _cb_model = CatBoostClassifier()
        _cb_model.load_model(str(CATBOOST_PATH))
    return _ft_model, _cb_model


def _texts_to_matrix(texts, ft_model):
    return np.vstack([ft_model.get_sentence_vector(str(t)) for t in texts]).astype(np.float32)


def predict_importance(text: str, threshold: float = None):
    threshold = 0.505
    CALIB_A = 6.9543863999053865
    CALIB_B = -3.9006695433483074

    ft_model, cb_model = load_models()
    X = _texts_to_matrix([text], ft_model)

    raw = float(cb_model.predict_proba(X)[0, 1])

    proba = 1.0 / (1.0 + np.exp(-(CALIB_A * raw + CALIB_B)))

    label = 1 if proba >= threshold else 0
    return label, proba
