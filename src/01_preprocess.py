"""
01_preprocess.py
Text normalization utilities for business name and address fields.
Language-agnostic: works for US, India, France, and any future country.
"""

import re
import unicodedata
import pandas as pd
from typing import Optional

# ─── Abbreviation expansion dictionaries ────────────────────────────────────
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
    r'\bnational\b': 'national',
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
    r'\bn\b': 'north',
    r'\bs\b': 'south',
    r'\be\b': 'east',
    r'\bw\b': 'west',
    r'\bne\b': 'northeast',
    r'\bnw\b': 'northwest',
    r'\bse\b': 'southeast',
    r'\bsw\b': 'southwest',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\bfl\b': 'floor',
    r'\bno\b': 'number',
    r'\b#\b': 'number',
}


def unicode_normalize(text: str) -> str:
    """
    Apply NFKD unicode normalization and re-encode to ASCII where possible.
    Keeps non-ASCII characters that don't have ASCII equivalents (e.g. Hindi script).
    This handles accents, ligatures, and some transliteration variants.
    """
    return unicodedata.normalize('NFKD', text)


def clean_text(text: Optional[str], abbrevs: dict = None) -> str:
    """
    Core text normalization function.
    Steps:
      1. Handle NaN / None → empty string
      2. Lowercase
      3. Unicode normalization (NFKD)
      4. Expand abbreviations
      5. Remove punctuation (keep alphanumeric + spaces)
      6. Collapse whitespace
    """
    if pd.isna(text) or text is None:
        return ''
    text = str(text).lower().strip()
    text = unicode_normalize(text)
    if abbrevs:
        for pattern, replacement in abbrevs.items():
            text = re.sub(pattern, replacement, text)
    # Remove punctuation but keep alphanumeric + space + unicode letters
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_name(name: Optional[str]) -> str:
    """Normalize a business name field."""
    return clean_text(name, abbrevs=NAME_ABBREVS)


def normalize_address(addr: Optional[str]) -> str:
    """Normalize a business address field."""
    return clean_text(addr, abbrevs=ADDR_ABBREVS)


def normalize_country(country: Optional[str]) -> str:
    """Lowercase country string — kept as open-set string label."""
    if pd.isna(country) or country is None:
        return ''
    return str(country).lower().strip()


def tokenize(text: str) -> list:
    """Split normalized text into word tokens, filter very short tokens."""
    return [t for t in text.split() if len(t) > 1]


def char_ngrams(text: str, n: int = 3) -> set:
    """Return the set of character n-grams from text."""
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all normalization to a source dataframe in-place (adds new columns).
    Returns the dataframe with added columns:
      - name_clean: normalized business_name
      - addr_clean: normalized business_address
      - country_clean: normalized country
      - name_tokens: list of word tokens from name_clean
      - addr_tokens: list of word tokens from addr_clean
    """
    df = df.copy()
    df['name_clean'] = df['business_name'].apply(normalize_name)
    df['addr_clean'] = df['business_address'].apply(normalize_address)
    df['country_clean'] = df['country'].apply(normalize_country)
    df['name_tokens'] = df['name_clean'].apply(tokenize)
    df['addr_tokens'] = df['addr_clean'].apply(tokenize)
    return df


if __name__ == '__main__':
    # Quick smoke test
    tests = [
        ("Pvt. EFS Print Ventures Ltd.", "name"),
        ("1795 Westchester Dr, High Point, NC", "addr"),
        ("Christ Chapel & Associates Corp.", "name"),
        (None, "name"),
    ]
    for text, kind in tests:
        if kind == 'name':
            result = normalize_name(text)
        else:
            result = normalize_address(text)
        print(f"  [{kind}] '{text}' → '{result}'")
    print("Preprocess OK")
