import re


def format_text_as_float(text: str) -> float | None:
    # Remove all characters that are not digits or decmial
    s = re.sub(pattern="[^0-9|.]", repl="", string=text)
    if s == "" or s == ".":
        return None
    else:
        return float(s)


def format_numeric_range(text: str) -> tuple[float, float]:
    "Match a numeric range with pattern like '30-60' and return values as tuple(30.0, 60.0)."
    pattern = r"(\d+)\s*-\s*(\d+)"
    s = re.search(pattern=pattern, string=text)
    if s is None:
        return None
    else:
        extracted = s.groups()
        # Check if extracted is a tuple
        if isinstance(extracted, tuple) and len(extracted) == 2:
            extracted = tuple([float(x) for x in extracted])
            return extracted
        else:
            return None


def format_text_as_int(text: str) -> int | None:
    # Remove all characters that are not digits
    if text.isdecimal():
        return int(text)
    else:
        s = re.sub(pattern="[^0-9]", repl="", string=text)
        if s == "":
            return None
        else:
            return int(s)


def format_text_as_bool(text: str) -> bool | None:
    # Match boolean text
    match = re.findall(pattern="True|true|False|false|Yes|yes|No|no", string=text)
    if match:
        # If multiple matched, only keep first
        if match[0] in ("True", "true", "Yes", "yes"):
            return True
        elif match[0] in ("False", "false", "No", "no"):
            return False
        else:
            return None
    else:
        return None


def format_text_as_string(text: str) -> str:
    # Strip Whitespace
    return str(text).strip()


def is_text_null(text: str) -> bool:
    text = text.lower()
    # If text contains null term, then we assume it is null
    pattern = "|".join(["none", "null", "na", "n/a", "nan", "nat"])
    match = re.search(pattern=pattern, string=text)
    if match is not None:
        return True
    else:
        return False


def format_asa(text: str) -> int | None:
    """Extract ASA-PS as integer. If null value, returns `None`.
    If numeric, then converts to int as ASA-PS."""
    if is_text_null(str(text)):
        return None
    # If numeric passed in, convert to int and accept if valid ASA-PS
    if isinstance(text, (int, float)):
        asa_ps = int(text)
        if asa_ps in (1, 2, 3, 4, 5, 6):
            return asa_ps
    # Otherwise, attempt to extract ASA-PS from string
    # Strip ASA-PS spelled out in different ways
    text = str(text).upper()
    asa_patterns = [
        "ASA PHYSICAL STATUS CLASSIFICATION",
        "ASA PHYSICAL STATUS",
        "ASA-PS CLASSIFICATION",
        "ASA PS CLASSIFICATION",
        "ASA-PS",
        "ASA PS",
        "ASA",
    ]
    for pattern in asa_patterns:
        if pattern in text:
            text = text.replace(pattern, "")

    # Strip all characters except for valid ones used in pattern matching
    text = re.sub(pattern="[^123456IiVv]", repl="", string=text)
    # Patterns for each ASA Class
    asa_1_match = re.findall(pattern=r"\b1\b|\bI\b|\bi\b", string=text)
    asa_2_match = re.findall(pattern=r"\b2\b|\bII\b|\bii\b", string=text)
    asa_3_match = re.findall(pattern=r"\b3\b|\bIII\b|\biii\b", string=text)
    asa_4_match = re.findall(pattern=r"\b4\b|\bIV\b|\biv\b", string=text)
    asa_5_match = re.findall(pattern=r"\b5\b|\bV\b|\bv\b", string=text)
    asa_6_match = re.findall(pattern=r"\b6\b|\bVI\b|\bvi\b", string=text)
    if asa_1_match:
        return 1
    elif asa_2_match:
        return 2
    elif asa_3_match:
        return 3
    elif asa_4_match:
        return 4
    elif asa_5_match:
        return 5
    elif asa_6_match:
        return 6
    else:
        return None


def resolve_numeric(text: str) -> float | None:
    "If null value given returns `None`, else convert to float."
    if is_text_null(str(text)):
        return None
    # Check if answer is a numeric range
    numeric_range = format_numeric_range(str(text))
    if numeric_range is not None:
        # numeric_range detected/extracted as tuple.  Get mean value in range.
        return float(sum(numeric_range) / 2)
    else:
        # Not a numeric range, assume single number
        if isinstance(text, float) or isinstance(text, int):
            return float(text)
        else:
            # Convert string to float or None
            return format_text_as_float(str(text))


def resolve_boolean(text: str) -> bool | None:
    "If null value given returns `None`, else convert to boolean."
    if is_text_null(str(text)):
        return None
    else:
        if isinstance(text, bool):
            return text
        else:
            # Convert string to bool or None
            return format_text_as_bool(text)
