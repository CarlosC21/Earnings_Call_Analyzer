# enhanced_parse_transcripts.py

import os
import re
import json
import spacy
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

# --- Config ---
INPUT_DIR = Path("data/transcripts")
OUTPUT_FILE = Path("data/parsed_transcripts.json")

# --- Helpers ---
def clean_text(text):
    return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

def remove_null_metrics(metrics_dict):
    """Remove metrics that are None instead of keeping them as null."""
    return {k: v for k, v in metrics_dict.items() if v is not None}

# --- Spoken number mapping ---
NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100
}

def spoken_to_number(text):
    """
    Convert spoken shorthand like:
      - 'three eighteen' -> 318
      - 'four twenty five' -> 425
      - 'one oh five' -> 105
    Supports millions/thousands/billions trailing words.
    """
    tokens = text.lower().split()
    digits = []
    for tok in tokens:
        if tok in NUM_WORDS and NUM_WORDS[tok] < 100:
            digits.append(str(NUM_WORDS[tok]))
        elif tok in ["oh", "o"]:
            digits.append("0")
        elif re.match(r"^\d+$", tok):
            digits.append(tok)
        else:
            break
    if digits:
        try:
            return int("".join(digits))
        except:
            return None
    return None

# --- Generic unit normalizer (returns millions) ---
def normalize_number(num_str, unit):
    try:
        num = float(num_str.replace(",", "").replace("$", ""))
    except Exception:
        return None
    u = (unit or "").lower().strip()
    if u in ["billion", "bn", "b"]:
        return round(num * 1000.0, 3)     # billions → millions
    elif u in ["million", "mn", "mm", "m"]:
        return round(num, 3)              # millions
    elif u in ["thousand", "k"]:
        return round(num / 1000.0, 3)     # thousands → millions
    else:
        if num >= 1_000_000:              # raw big number → millions
            return round(num / 1_000_000.0, 3)
        return round(num, 3)

# --- Metric parsers ---
def parse_metric(text, keyword):
    """
    Generic parser: find '<keyword> ... $<number> <unit?>'
    Uses a non-greedy gap and escapes the keyword to avoid regex pitfalls.
    """
    t = clean_text(text)
    pattern = re.compile(
        rf"{re.escape(keyword)}(?:\s+\w+)*?\s*"
        rf"(?:was|were|of|at|totaled|amounted to|came in at|reported at|reached)?\s*"
        rf"(?:approximately|about|around|nearly|over|~)?\s*"
        rf"\$?\s*([\d,\.]+)(?:\s*(billion|million|thousand|k|bn|mn|mm|m|b))?",
        re.IGNORECASE,
    )
    match = pattern.search(t)
    if match:
        return normalize_number(match.group(1), match.group(2))
    return None

def parse_revenue(text):
    """
    Extract revenue in millions robustly.
    Handles:
      - '$14,400,000,000'
      - '14.4 billion'
      - 'Revenue of $14,400,000,000'
      - 'Revenue reached $14.4B'
    """
    t = clean_text(text)

    # Regex: look for 'revenue' followed by optional words and number/unit
    pattern = re.compile(
        r"\b(revenue)\b(?:\s+\w+){0,5}?"           # 'revenue' + optional words
        r"(?:was|were|of|at|totaled|amounted to|came in at|reported at|reached)?\s*"
        r"(?:approximately|about|around|nearly|over|~)?\s*"
        r"\$?\s*([\d,]+(?:\.\d+)?)\s*"            # number with commas/decimal
        r"(billion|million|thousand|k|bn|mn|mm|m|b)?",  # optional unit
        re.IGNORECASE
    )

    matches = pattern.findall(t)
    for _, number, unit in matches:
        val = normalize_number(number, unit)
        if val is not None:
            return val

    # fallback: capture any large $ number if 'revenue' appears anywhere
    fallback_pattern = re.compile(
        r"\$([\d,]{7,})(?:\.\d+)?", re.IGNORECASE
    )
    if "revenue" in t.lower():
        fallback_match = fallback_pattern.search(t)
        if fallback_match:
            return normalize_number(fallback_match.group(1), None)

    return None

def parse_consolidated_net_sales(text): return parse_metric(text, "consolidated net sales")
def parse_consolidated_operating_income(text): return parse_metric(text, "consolidated operating income")
def parse_marketplace_gov(text): return parse_metric(text, "marketplace gov")
def parse_total_orders(text): return parse_metric(text, "total orders")
def parse_gaap_net_income(text): return parse_metric(text, "gaap net income")

# --- NET INCOME (NEW & ROBUST) ---
def parse_net_income(text):
    """
    Extract net income or net loss in millions.
    Returns negative value if it's a net loss.
    Excludes adjusted net income (handled separately).
    """
    t = clean_text(text)
    doc = nlp(t)

    # skip adjusted mentions
    sentences = [s for s in doc.sents if "adjusted net income" not in s.text.lower()]

    pattern = re.compile(
        r"\bnet\s+(income|earnings|profit|loss)\b"
        r"(?:\s+\w+){0,6}?\s*"
        r"(?:was|were|of|at|totaled|amounted to|came in at|reported at|came to|reached)?\s*"
        r"(?:approximately|about|around|nearly|over|~)?\s*"
        r"\$?\s*([\d,]+(?:\.\d+)?)\s*"
        r"(billion|million|thousand|bn|mn|mm|m|b|k)?",
        re.IGNORECASE
    )

    for sent in sentences:
        s = sent.text
        m = pattern.search(s)
        if m:
            val = normalize_number(m.group(2).replace(",", ""), m.group(3))
            if val is not None and m.group(1).lower() == "loss":
                val = -val
            return val

    return None

def parse_adjusted_ebitda(text):
    """
    Extract adjusted EBITDA in millions from sentences like:
      - "adjusted EBITDA was over $300,000,000"
      - "$300,000,000 adjusted EBITDA"
      - "adjusted EBITDA of $0.3 billion"
    Robustly stops at the first number after the keyword.
    """
    t = clean_text(text)

    # Number immediately after the keyword
    pattern_after = re.compile(
        r"adjusted\s+ebitda(?:\s+\w+){0,3}?\s*"
        r"(?:was|were|of|at|totaled|amounted to|came in at|reported at|came to|reached)?\s*"
        r"(?:approximately|about|around|nearly|over|~)?\s*"
        r"\$?\s*([\d,]+(?:\.\d+)?)\s*"
        r"(billion|million|thousand|k|bn|mn|mm|m|b)?",
        re.IGNORECASE
    )

    match = pattern_after.search(t)
    if match:
        return normalize_number(match.group(1).replace(",", ""), match.group(2))

    # Number immediately before the keyword
    pattern_before = re.compile(
        r"\$?\s*([\d,]+(?:\.\d+)?)\s*"
        r"(billion|million|thousand|k|bn|mn|mm|m|b)?\s+adjusted\s+ebitda",
        re.IGNORECASE
    )
    match = pattern_before.search(t)
    if match:
        return normalize_number(match.group(1).replace(",", ""), match.group(2))

    return None

def parse_available_liquidity(text):
    val = parse_metric(text, "available liquidity")
    if val is not None:
        return val
    t = clean_text(text)
    pattern = re.compile(
        r"\$?\s*([\d,\.]+)\s*(billion|million|thousand|k|bn|mn|mm|m|b)?\s+(?:in|of|was)\s+available\s+liquidity",
        re.IGNORECASE
    )
    match = pattern.search(t)
    if match:
        return normalize_number(match.group(1), match.group(2))
    return None

def parse_net_cash_balance(text):
    t = clean_text(text)
    pattern = re.compile(
        r"net cash balance\s+(?:at|of|was)?\s*\$?\s*([\d,\.]+)\s*(billion|million|thousand|k|bn|mn|mm|m|b)?",
        re.IGNORECASE
    )
    match = pattern.search(t)
    if match:
        return normalize_number(match.group(1), match.group(2))
    return None

def parse_credit_facility(text):
    t = clean_text(text)
    pattern = re.compile(
        r"\$?\s*([\d,\.]+)\s*(billion|million|thousand|bn|mn|mm|m|b|k)?\s+(?:credit facility)",
        re.IGNORECASE
    )
    match = pattern.search(t)
    if match:
        return normalize_number(match.group(1), match.group(2))
    return None

# --- Share buybacks / dividends ---
def parse_share_buybacks_and_dividends(text):
    t = clean_text(text)
    pattern = re.compile(
        r"(?:(?:nearly|approximately|~|about|around|over)?\s*\$?\s*([\d,]+(?:\.\d+)?)(?:\s*(billion|million|thousand|k|bn|mn|mm|m|b))?)"
        r"(?:\s*(?:of|in|towards|for|used for|was|were|deployed towards)?\s*(?:\w+\s*){0,6})"
        r"(share buybacks|share repurchases|dividends)",
        re.IGNORECASE
    )
    match = pattern.search(t)
    if match:
        return normalize_number(match.group(1), match.group(2))

    pattern_rev = re.compile(
        r"(share buybacks|share repurchases|dividends)"
        r"(?:\s*(?:of|in|towards|for|used for|was|were|deployed towards)?\s*(?:\w+\s*){0,6})"
        r"\$?\s*([\d,]+(?:\.\d+)?)(?:\s*(billion|million|thousand|k|bn|mn|mm|m|b))?",
        re.IGNORECASE
    )
    match = pattern_rev.search(t)
    if match:
        return normalize_number(match.group(2), match.group(3))
    return None

def parse_combined_rate_ar6(text):
    t = clean_text(text)
    pattern = re.compile(
        r"preliminary combined rate for AR6\s*(?:is|of|was)?\s*([\d,.]+)\s*%",
        re.IGNORECASE
    )
    match = pattern.search(t)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None

def parse_lumber_segment_adjusted_ebitda(text):
    t = clean_text(text)
    pattern = re.compile(
        r"lumber segment.*?adjusted ebitda\s*(?:of|was)?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|k|bn|mn|mm|m|b)?",
        re.IGNORECASE
    )
    match = pattern.search(t)
    if match:
        return normalize_number(match.group(1), match.group(2))
    return None

# --- Other metrics ---
def parse_orders(text):
    t = clean_text(text)
    doc = nlp(t)
    for sent in doc.sents:
        if "orders" in sent.text.lower():
            pattern = re.compile(r"orders\s+(?:of|were|was|totaled|amounted to)?\s*\$?([\d,.]+)", re.IGNORECASE)
            match = pattern.search(sent.text)
            if match:
                return normalize_number(match.group(1), None)
    return None

def parse_cash_flow(text):
    """
    Extract operating/overall cash flow, excluding free cash flow (handled separately).
    """
    t = clean_text(text)
    doc = nlp(t)
    for sent in doc.sents:
        s = sent.text.lower()
        # Skip if it's free cash flow (already handled)
        if "free cash flow" in s:
            continue
        if "cash flow" in s or "cash from operations" in s:
            pattern = re.compile(
                r"(?:generated|was|were|of|totaled|amounted to)\s*\$?([\d,.]+)",
                re.IGNORECASE
            )
            match = pattern.search(sent.text)
            if match:
                return normalize_number(match.group(1), None)
    return None

def parse_gaap_diluted_eps(text):
    t = clean_text(text)
    pattern = re.compile(r"gaap diluted eps.*?\$?\s*([\d,\.-]+)", re.IGNORECASE)
    match = pattern.search(t)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except:
            return None
    return None

def parse_adjusted_eps(text):
    t = clean_text(text)
    doc = nlp(t)
    for sent in doc.sents:
        if "adjusted" in sent.text.lower() and "eps" in sent.text.lower():
            pattern = re.compile(r"\b(?:was|were|of|at|totaled|amounted to)\s*\$?(-?[\d]+\.[\d]+)", re.IGNORECASE)
            match = pattern.search(sent.text)
            if match:
                try:
                    return float(match.group(1))
                except:
                    return None
    return None

def parse_actual_eps(text):
    """
    Extract actual EPS values.
    Always positive unless the surrounding text explicitly indicates a loss/negative.
    Handles formats like:
      - 'Actual EPS $0.17'
      - 'Actual EPS: -0.25 (loss)'
      - 'The actual EPS was 17 cents'
    """
    t = clean_text(text)

    # broader regex: allow optional colon, space, $, or no space at all
    pattern = re.compile(r"actual\s+eps[:\s]*\$?\s*(-?[\d,.]+)", re.IGNORECASE)
    match = pattern.search(t)
    if match:
        try:
            eps = float(match.group(1).replace(",", ""))
            eps = abs(eps)
            if re.search(r"\b(loss|negative|deficit)\b", t, re.IGNORECASE):
                eps = -eps
            return round(eps, 3)
        except ValueError:
            return None

    # also handle spoken formats: "Actual EPS of seventeen cents"
    spoken_pattern = re.compile(r"actual\s+eps\s+(?:of|was|were|at|totaled)?\s*([a-z\s]+)\s*(cents?|dollars?)", re.IGNORECASE)
    match = spoken_pattern.search(t)
    if match:
        num_val = spoken_to_number(match.group(1))
        if num_val is not None:
            val = float(num_val)
            if "cent" in match.group(2).lower():
                val /= 100.0
            if re.search(r"\b(loss|negative|deficit)\b", t, re.IGNORECASE):
                val = -val
            return round(val, 3)

    return None

def parse_net_revenue_margin(text):
    t = clean_text(text)
    pattern = re.compile(r"net revenue margin.*?([\d,.]+)\s*%", re.IGNORECASE)
    match = pattern.search(t)
    if match:
        return float(match.group(1).replace(",", ""))
    return None

# --- Metric parser for total net sales ---
def parse_total_net_sales(text):
    """
    Extract total net sales in millions.
    Only considers numbers that are clearly monetary amounts.
    """
    t = clean_text(text)

    # Look for "total net sales" followed by optional words, then a number with $ or commas/decimals
    pattern = re.compile(
        r"total\s+net\s+sales(?:\s+\w+){0,5}?"      # optional words after keyword
        r"\s*(?:was|were|of|at|totaled|amounted to|came in at|reported at|reached)?\s*"
        r"(?:approximately|about|around|nearly|over|~)?\s*"
        r"\$"                                        # must have a dollar sign
        r"\s*([\d,]+(?:\.\d+)?)\s*"                 # number with commas/decimal
        r"(billion|million|thousand|k|bn|mn|mm|m|b)?",
        re.IGNORECASE,
    )

    match = pattern.search(t)
    if match:
        return normalize_number(match.group(1), match.group(2))

    # fallback: numbers like "1.48 billion" even without $
    pattern2 = re.compile(
        r"total\s+net\s+sales(?:\s+\w+){0,5}?\s*"
        r"(?:was|were|of|at|totaled|amounted to|came in at|reported at|reached)?\s*"
        r"(?:approximately|about|around|nearly|over|~)?\s*"
        r"([\d,.]+)\s*(billion|million|thousand|k|bn|mn|mm|m|b)",
        re.IGNORECASE,
    )
    match = pattern2.search(t)
    if match:
        return normalize_number(match.group(1), match.group(2))

    return None

def parse_liquidity(text):
    """
    Extract liquidity in millions.
    Prioritizes 'total liquidity' mentions over incremental changes.
    """
    t = clean_text(text)

    # Case 1: explicit "total liquidity"
    pattern_total = re.compile(
        r"\btotal\s+liquidity(?:\s+\w+){0,3}?\s*"
        r"(?:was|were|of|at|totaled|amounted to|reported at|came in at|reached|stood at)?\s*"
        r"\$?\s*([\d,\.]+)\s*"
        r"(billion|million|thousand|bn|mn|mm|m|b|k)\b",
        re.IGNORECASE
    )
    m = pattern_total.search(t)
    if m:
        return normalize_number(m.group(1), m.group(2))

    # Case 2: number before "liquidity"
    pattern_before = re.compile(
        r"\$?\s*([\d,\.]+)\s*"
        r"(billion|million|thousand|bn|mn|mm|m|b|k)\s+"
        r"(?:in|of|at|was|for|representing)?\s*liquidity",
        re.IGNORECASE
    )
    m = pattern_before.search(t)
    if m:
        return normalize_number(m.group(1), m.group(2))

    # Case 3: number after "liquidity" (only if tied to state, not "by" increments)
    pattern_after = re.compile(
        r"\bliquidity(?:\s+\w+){0,3}?\s*"
        r"(?:was|were|of|at|totaled|amounted to|reported at|came in at|stood at|reached)?\s*"
        r"\$?\s*([\d,\.]+)\s*"
        r"(billion|million|thousand|bn|mn|mm|m|b|k)\b",
        re.IGNORECASE
    )
    m = pattern_after.search(t)
    if m:
        return normalize_number(m.group(1), m.group(2))

    return None

def parse_free_cash_flow(text):
    """
    Robustly extract free cash flow in millions.
    Captures the number immediately after the phrase 'free cash flow'.
    """
    # Keep original text, do NOT clean spaces
    t = text

    # Strict regex: free cash flow immediately followed by optional 'of/was' and the number
    pattern = re.compile(
        r"free\s+cash\s+flow\s*(?:of|was|were)?\s*\$?\s*([\d]{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
        r"(billion|million|thousand|k|bn|mn|mm|m|b)?",
        re.IGNORECASE
    )

    match = pattern.search(t)
    if match:
        number_str = match.group(1)
        unit = match.group(2)
        return normalize_number(number_str, unit)

    return None

def parse_interest_expense(text):
    """
    Extract interest expense in millions.
    Handles:
      - "interest expense to $180,000,000"
      - "interest expense was $180 million"
      - "$0.18 billion of interest expense"
    """
    t = clean_text(text)

    # Case 1: number after "interest expense"
    pattern_after = re.compile(
        r"interest\s+expense(?:\s+\w+){0,5}?\s*"
        r"(?:was|were|is|of|at|to|for|totaled|amounted to|came in at|reported at|reached)?\s*"
        r"(?:approximately|about|around|nearly|over|~)?\s*"
        r"\$?\s*([\d,\.]+)\s*"
        r"(billion|million|thousand|k|bn|mn|mm|m|b)?",
        re.IGNORECASE
    )
    m = pattern_after.search(t)
    if m:
        return normalize_number(m.group(1), m.group(2))

    # Case 2: number before "interest expense"
    pattern_before = re.compile(
        r"\$?\s*([\d,\.]+)\s*"
        r"(billion|million|thousand|k|bn|mn|mm|m|b)?"
        r"(?:\s+\w+){0,5}?\s+interest\s+expense",
        re.IGNORECASE
    )
    m = pattern_before.search(t)
    if m:
        return normalize_number(m.group(1), m.group(2))

    return None

def parse_adjusted_net_income(text):
    """
    Extract adjusted net income in millions.
    Handles:
      - "adjusted net income was $15,000,000"
      - "$15 million in adjusted net income"
      - "adjusted net income of $0.015 billion"
    """
    t = clean_text(text)

    # Case 1: number BEFORE keyword (most common, prevents picking up years after)
    pattern_before = re.compile(
        r"\$?\s*([\d]{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
        r"(billion|million|thousand|bn|mn|mm|m|b|k)?"
        r"(?:\s+\w+){0,3}?\s+adjusted\s+net\s+income\b",
        re.IGNORECASE,
    )
    m = pattern_before.search(t)
    if m:
        return normalize_number(m.group(1), m.group(2))

    # Case 2: number AFTER keyword (stricter – requires $ or unit to avoid years)
    pattern_after = re.compile(
        r"\badjusted\s+net\s+income(?:\s+\w+){0,3}?\s*"
        r"(?:was|were|is|of|at|totaled|amounted to|came in at|reported at|reached|came to)?\s*"
        r"(?:approximately|about|around|nearly|over|~)?\s*"
        r"\$?\s*([\d]{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
        r"(billion|million|thousand|bn|mn|mm|m|b|k)\b",  # 🔒 unit REQUIRED here
        re.IGNORECASE,
    )
    m = pattern_after.search(t)
    if m:
        return normalize_number(m.group(1), m.group(2))

    return None

def parse_net_income_margin(text):
    """
    Robust extraction of net income margin percentages from complex sentences.
    Handles patterns like:
    - 'net income margin of 6.4%'
    - 'net income margin increased five points to 6.4%'
    - 'Net income margin, on a GAAP basis, was 6.4%'
    Returns float or None.
    """
    t = clean_text(text).lower()

    # Find the start of 'net income margin'
    keyword_pos = t.find("net income margin")
    if keyword_pos == -1:
        return None

    # Slice the text from the keyword onwards
    snippet = t[keyword_pos : keyword_pos + 200]  # take next 200 chars to cover long phrases

    # Look for the first percentage number in this snippet
    match = re.search(r"([\d,.]+)\s*%", snippet)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None

def parse_gaap_operating_income(text):
    """
    Extract GAAP operating income in millions robustly.
    Handles:
      - '$500,000,000 of quarterly GAAP operating income'
      - 'GAAP operating income of $0.5 billion'
      - 'quarterly GAAP operating income was $500 million'
      - '$500M GAAP operating income'
      - '$500 thousand GAAP operating income'
    """
    t = clean_text(text)

    # Regex: look for a number (with optional commas/decimal) near 'GAAP operating income'
    pattern = re.compile(
        r"(\$?[\d,.]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|bn|mn|mm|m|b|k))?)"
        r"(?:\s*(?:of|in|for|quarterly|this quarter)?)\s*gaap operating income",
        re.IGNORECASE
    )

    match = pattern.search(t)
    if match:
        # Extract number and unit separately
        num_unit = match.group(1)
        num_match = re.match(r"\$?([\d,.]+)", num_unit)
        unit_match = re.search(r"(billion|million|thousand|bn|mn|mm|m|b|k)", num_unit, re.IGNORECASE)

        number = num_match.group(1) if num_match else None
        unit = unit_match.group(1) if unit_match else None
        if number:
            return normalize_number(number, unit)

    # Fallback: check if GAAP operating income comes first
    pattern_after = re.compile(
        r"gaap operating income\s*(?:was|were|of|at|totaled|amounted to|came in at|reported at|reached)?\s*(\$?[\d,.]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|bn|mn|mm|m|b|k))?)",
        re.IGNORECASE
    )
    match = pattern_after.search(t)
    if match:
        num_unit = match.group(1)
        num_match = re.match(r"\$?([\d,.]+)", num_unit)
        unit_match = re.search(r"(billion|million|thousand|bn|mn|mm|m|b|k)", num_unit, re.IGNORECASE)
        number = num_match.group(1) if num_match else None
        unit = unit_match.group(1) if unit_match else None
        if number:
            return normalize_number(number, unit)

    return None

def parse_total_available_liquidity(text):
    """
    Extract total available liquidity in millions.
    Handles formats like:
      - "$12,000,000,000 of total available liquidity"
      - "total available liquidity of $12 billion"
    """
    t = clean_text(text)

    # Case 1: number BEFORE keyword
    pattern_before = re.compile(
        r"\$?\s*([\d,\.]+)\s*(billion|million|thousand|k|bn|mn|mm|m|b)?\s+of\s+total\s+available\s+liquidity",
        re.IGNORECASE
    )
    m = pattern_before.search(t)
    if m:
        return normalize_number(m.group(1), m.group(2))

    # Case 2: number AFTER keyword
    pattern_after = re.compile(
        r"total\s+available\s+liquidity\s*(?:was|were|of|at|totaled|amounted to|reported at|came in at|stood at|reached)?\s*\$?\s*([\d,\.]+)\s*(billion|million|thousand|k|bn|mn|mm|m|b)?",
        re.IGNORECASE
    )
    m = pattern_after.search(t)
    if m:
        return normalize_number(m.group(1), m.group(2))

    # fallback: look for "available liquidity" anywhere
    return parse_available_liquidity(text)

def extract_demand_sentences(text):
    """
    Extract sentences mentioning demand.
    Examples:
      - "We saw strong demand across all categories."
      - "Customer demand softened in Q2."
      - "Demand for our products remains resilient."
    """
    doc = nlp(clean_text(text))
    return [s.text.strip() for s in doc.sents if "demand" in s.text.lower()]

# --- Contextual extractions ---
def extract_forward_looking_statements(text):
    doc = nlp(clean_text(text))
    phrases = ["we expect", "we anticipate", "looking ahead", "we forecast", "we believe", "we see", "we project"]
    return [s.text.strip() for s in doc.sents if any(p in s.text.lower() for p in phrases)]

def extract_prior_year_comparisons(text):
    doc = nlp(clean_text(text))
    return [s.text.strip() for s in doc.sents if any(x in s.text.lower() for x in ["prior year", "last year", "last year's"])]

def extract_year_over_year_sentences(text):
    """
    Extract sentences mentioning 'year over year' or 'year-over-year'.
    """
    doc = nlp(clean_text(text))
    pattern = re.compile(r"year[-\s]?over[-\s]?year", re.IGNORECASE)
    return [s.text.strip() for s in doc.sents if pattern.search(s.text)]

def extract_guidance_sentences(text):
    doc = nlp(clean_text(text))
    return [s.text.strip() for s in doc.sents if "guidance" in s.text.lower()]

def extract_compared_to_sentences(text):
    doc = nlp(clean_text(text))
    return [s.text.strip() for s in doc.sents if "compared to" in s.text.lower()]

def extract_raising_sentences(text):
    doc = nlp(clean_text(text))
    # Match raise, raised, raising
    return [s.text.strip() for s in doc.sents if re.search(r"\brais(?:e|ed|ing)\b", s.text, re.IGNORECASE)]

# --- Refinance / Refinanced sentences ---
def extract_refinance_sentences(text):
    """
    Extract sentences mentioning 'refinance' or 'refinanced'.
    """
    doc = nlp(clean_text(text))
    return [s.text.strip() for s in doc.sents if re.search(r"\brefinanc(?:e|ed)\b", s.text, re.IGNORECASE)]

# --- Production sentences ---
def extract_production_sentences(text):
    """
    Extract sentences that mention production.
    Examples:
      - "Our production increased 5% this quarter."
      - "We saw record oil production in Q2."
    """
    doc = nlp(clean_text(text))
    return [s.text.strip() for s in doc.sents if "production" in s.text.lower()]

def extract_outlook_sentences(text):
    """
    Extract sentences mentioning 'outlook', 'expectation(s)', or 'forecast' in the context of future performance.
    """
    doc = nlp(clean_text(text))
    keywords = ["outlook", "expectation", "expectations", "forecast"]
    return [s.text.strip() for s in doc.sents if any(k in s.text.lower() for k in keywords)]

def extract_challenge_sentences(text):
    """
    Extract sentences mentioning 'challenge', 'challenged', or 'challenging'.
    Examples:
      - "We faced significant challenges in Q2."
      - "Our supply chain was challenged by inflation."
      - "The team is challenging prior assumptions."
    """
    doc = nlp(clean_text(text))
    return [s.text.strip() for s in doc.sents if re.search(r"\bchalleng(?:e|ed|ing|es)\b", s.text, re.IGNORECASE)]

# --- Metrics Extraction ---
def extract_metrics(text):
    results = {}
    results["revenue"] = parse_revenue(text)
    results["consolidated_net_sales"] = parse_consolidated_net_sales(text)
    results["consolidated_operating_income"] = parse_consolidated_operating_income(text)
    results["orders"] = parse_orders(text)
    results["total_orders"] = parse_total_orders(text)
    results["marketplace_gov"] = parse_marketplace_gov(text)
    results["adjusted_ebitda"] = parse_adjusted_ebitda(text)
    results["lumber_segment_adjusted_ebitda"] = parse_lumber_segment_adjusted_ebitda(text)
    results["available_liquidity"] = parse_available_liquidity(text)
    results["liquidity"] = parse_liquidity(text)   # <-- NEW
    results["net_cash_balance"] = parse_net_cash_balance(text)
    results["credit_facility"] = parse_credit_facility(text)
    results["cash_flow"] = parse_cash_flow(text)
    results["net_revenue_margin"] = parse_net_revenue_margin(text)
    results["gaap_net_income"] = parse_gaap_net_income(text)
    results["net_income"] = parse_net_income(text)   # <-- NEW
    results["gaap_diluted_eps"] = parse_gaap_diluted_eps(text)
    results["adjusted_eps"] = parse_adjusted_eps(text)
    results["actual_eps"] = parse_actual_eps(text)
    results["share_buybacks_and_dividends"] = parse_share_buybacks_and_dividends(text)
    results["combined_rate_ar6"] = parse_combined_rate_ar6(text)
    results["total_net_sales"] = parse_total_net_sales(text)
    results["free_cash_flow"] = parse_free_cash_flow(text)
    results["interest_expense"] = parse_interest_expense(text)
    results["adjusted_net_income"] = parse_adjusted_net_income(text)
    results["gaap_operating_income"] = parse_gaap_operating_income(text)
    results["total_available_liquidity"] = parse_total_available_liquidity(text)


    results = remove_null_metrics(results)

    results["guidance"] = extract_guidance_sentences(text)
    results["forward_look"] = extract_forward_looking_statements(text)
    results["prior_year_mentions"] = extract_prior_year_comparisons(text)
    results["year_over_year_mentions"] = extract_year_over_year_sentences(text)
    results["compared_to_mentions"] = extract_compared_to_sentences(text)
    results["raising_mentions"] = extract_raising_sentences(text)
    results["refinance_mentions"] = extract_refinance_sentences(text)
    results["production"] = extract_production_sentences(text)   # <-- NEW
    results["outlook"] = extract_outlook_sentences(text)   # <-- NEW
    results["demand_mentions"] = extract_demand_sentences(text)
    results["challenge_mentions"] = extract_challenge_sentences(text)

    return results

# --- Ticker / Quarter / Date Extraction ---
def extract_ticker_quarter_year_date(text):
    first_lines = "\n".join(text.split("\n")[:3])
    ticker_match = re.search(r"\(([A-Z]+):([A-Z]+)\)", first_lines)
    quarter_year_match = re.search(r"\b(Q[1-4])\s+(\d{4})\b", first_lines)
    date_match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        first_lines,
    )
    ticker = ticker_match.group(2) if ticker_match else None
    quarter = quarter_year_match.group(1) if quarter_year_match else None
    year = quarter_year_match.group(2) if quarter_year_match else None
    earnings_date = date_match.group(0) if date_match else None
    return ticker, quarter, year, earnings_date

def extract_from_filename(file_name):
    name = Path(file_name).stem
    parts = name.split("_")
    ticker = parts[0].upper() if len(parts) > 0 else None
    quarter = parts[1] if len(parts) > 1 else None
    year = None
    earnings_date = None
    if len(parts) >= 4:
        if re.match(r"\d{4}", parts[-1]):
            year = parts[-1]
            month_day = " ".join(parts[2:-1])
            earnings_date = f"{month_day} {year}"
    return ticker, quarter, year, earnings_date

def parse_transcript_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    ticker, quarter, year, earnings_date = extract_ticker_quarter_year_date(text)
    if not ticker or not quarter or not year or not earnings_date:
        ticker, quarter, year, earnings_date = extract_from_filename(file_path.name)
    metrics = extract_metrics(text)
    return {
        "ticker": ticker,
        "quarter": quarter,
        "year": year,
        "earnings_date": earnings_date,
        "file_name": file_path.name,
        "metrics": metrics,
    }

def main():
    parsed_results = []
    for file_path in INPUT_DIR.glob("*.txt"):
        parsed_results.append(parse_transcript_file(file_path))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(parsed_results, out, indent=2)
    print(f"Parsed {len(parsed_results)} transcript(s) into {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
