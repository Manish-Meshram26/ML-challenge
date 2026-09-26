"""
preprocess_utils.py
Standalone preprocessing utilities importable from any module.
(Separate from 01_preprocess.py to avoid circular import issues.)
"""

import re
import unicodedata
import pandas as pd
from typing import Optional

NAME_ABBREVS = {
    r'\bcorp\b': 'corporation',
    r'\bpvt\b': 'private',
    r'\bltd\b': 'limited',
    r'\binc\b': 'incorporated',
    r'\bllc\b': 'limited liability company',
    r'\bllp\b': 'limited liability partnership',
    r'\bco\b': 'company',
    r'\b&\b': 'and',
    r'\bgrp\b': 'group',
    r'\bintl\b': 'international',
    r'\bsvcs\b': 'services',
    r'\bsvc\b': 'service',
    r'\bmgmt\b': 'management',
    r'\bassc\b': 'associates',
    r'\bassoc\b': 'associates',
    r'\bentps\b': 'enterprises',
    r'\bentp\b': 'enterprise',
    r'\btech\b': 'technology',
    r'\bmfg\b': 'manufacturing',
    r'\bdist\b': 'distributors',
    r'\bnatl\b': 'national',
    r'\bdev\b': 'development',
    r'\binds\b': 'industries',
    r'\bind\b': 'industries',
}

ADDR_ABBREVS = {
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bave\b': 'avenue',
    r'\bav\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bfwy\b': 'freeway',
    r'\bhwy\b': 'highway',
    r'\bpkwy\b': 'parkway',
    r'\bsq\b': 'square',
    r'\bexpy\b': 'expressway',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\bfl\b': 'floor',
    r'\bno\b': 'number',
    r'\b#\b': 'number',
}


def unicode_normalize(text: str) -> str:
    return unicodedata.normalize('NFKD', text)


def clean_text(text: Optional[str], abbrevs: dict = None) -> str:
    if pd.isna(text) or text is None:
        return ''
    text = str(text).lower().strip()
    text = unicode_normalize(text)
    if abbrevs:
        for pattern, replacement in abbrevs.items():
            text = re.sub(pattern, replacement, text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_name(name: Optional[str]) -> str:
    return clean_text(name, abbrevs=NAME_ABBREVS)


def normalize_address(addr: Optional[str]) -> str:
    return clean_text(addr, abbrevs=ADDR_ABBREVS)


def normalize_country(country: Optional[str]) -> str:
    if pd.isna(country) or country is None:
        return ''
    return str(country).lower().strip()


def tokenize(text: str) -> list:
    return [t for t in text.split() if len(t) > 1]


def char_ngrams(text: str, n: int = 3) -> set:
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all normalization to a source dataframe.
    Adds: name_clean, addr_clean, country_clean, name_tokens, addr_tokens
    """
    df = df.copy()
    df['name_clean'] = df['business_name'].apply(normalize_name)
    df['addr_clean'] = df['business_address'].apply(normalize_address)
    df['country_clean'] = df['country'].apply(normalize_country)
    df['name_tokens'] = df['name_clean'].apply(tokenize)
    df['addr_tokens'] = df['addr_clean'].apply(tokenize)
    return df
