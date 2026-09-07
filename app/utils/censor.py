"""Profanity and sensitive content censorship filter.

Masks swear words, explicit anatomical terms, and slurs with asterisks
to protect YouTube video monetization, prevent age-restrictions, and maintain
broad algorithmic distribution in the Shorts feed.
"""
from __future__ import annotations

import re
from typing import Sequence

# Map of lowercase root swear/profane words to their masked equivalent
_PROFANITY_MAP: dict[str, str] = {
    # F-words
    "fuck": "f*ck",
    "fucker": "f*cker",
    "fuckers": "f*ckers",
    "fucking": "f*cking",
    "fucked": "f*cked",
    "fucks": "f*cks",
    "motherfucker": "motherf*cker",
    "motherfucking": "motherf*cking",
    # S-words
    "shit": "sh*t",
    "shits": "sh*ts",
    "shitting": "sh*tting",
    "shitty": "sh*tty",
    "bullshit": "bullsh*t",
    "dipshit": "dipsh*t",
    "horseshit": "horsesh*t",
    # B-words
    "bitch": "b*tch",
    "bitches": "b*tches",
    "bitching": "b*tching",
    "bitchy": "b*tchy",
    "bastard": "b*stard",
    "bastards": "b*stards",
    # A-words
    "asshole": "a**hole",
    "assholes": "a**holes",
    "dumbass": "dumba**",
    "jackass": "jacka**",
    "badass": "bada**",
    "ass": "a**",
    "asses": "a**es",
    # C/D-words & Anatomy
    "cock": "c*ck",
    "cocks": "c*cks",
    "cocksucker": "c*cksucker",
    "dick": "d*ck",
    "dicks": "d*cks",
    "dickhead": "d*ckhead",
    "pussy": "p*ssy",
    "pussies": "p*ssies",
    "cunt": "c*nt",
    "cunts": "c*nts",
    "tits": "t*ts",
    "titties": "t*tties",
    # Explicit / Slurs
    "nigga": "n***a",
    "niggas": "n***as",
    "nigger": "n***er",
    "niggers": "n***ers",
    "whore": "wh*re",
    "whores": "wh*res",
    "slut": "sl*t",
    "sluts": "sl*ts",
    "porn": "p*rn",
    "porno": "p*rno",
    "sex": "s*x",
    "sexy": "s*xy",
    "nude": "n*de",
    "nudes": "n*des",
    "naked": "n*ked",
    "rape": "r*pe",
    "raped": "r*ped",
    "rapist": "r*pist",
    "suicide": "s*icide",
    "cocaine": "c*caine",
}


def _match_casing(original: str, masked: str) -> str:
    """Preserve the uppercase/title/lowercase style of the original word."""
    if original.isupper():
        return masked.upper()
    if original.istitle():
        return masked.capitalize()
    return masked.lower()


def censor_word(raw_word: str) -> str:
    """Censor a single word while preserving leading/trailing punctuation and casing.

    Examples:
        'fuck' -> 'f*ck'
        'FUCKING!' -> 'F*CKING!'
        '\"bitch,\"' -> '\"b*tch,\"'
    """
    if not raw_word:
        return raw_word

    # Extract leading punctuation, core letters/numbers, and trailing punctuation
    match = re.match(r"^([^a-zA-Z0-9]*)([a-zA-Z0-9]+)([^a-zA-Z0-9]*)$", raw_word)
    if not match:
        lower = raw_word.lower()
        if lower in _PROFANITY_MAP:
            return _match_casing(raw_word, _PROFANITY_MAP[lower])
        return raw_word

    prefix, core, suffix = match.groups()
    lower_core = core.lower()

    if lower_core in _PROFANITY_MAP:
        masked_core = _match_casing(core, _PROFANITY_MAP[lower_core])
        return f"{prefix}{masked_core}{suffix}"

    return raw_word


def censor_text(text: str) -> str:
    """Censor all profanities in a sentence, title, or paragraph while preserving whitespace."""
    if not text:
        return text

    tokens = re.split(r"(\s+)", text)
    result = []
    for token in tokens:
        if token.isspace() or not token:
            result.append(token)
        else:
            result.append(censor_word(token))
    return "".join(result)


def censor_word_tuples(
    words: Sequence[tuple[str, float, float]]
) -> list[tuple[str, float, float]]:
    """Censor words in a list of (word, start_time, end_time) tuples for subtitle rendering."""
    censored = []
    for word, start, end in words:
        new_word = censor_word(word)
        censored.append((new_word, start, end))
    return censored
