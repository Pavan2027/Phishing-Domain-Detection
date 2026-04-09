# features/lexical.py
import re
import math
import tldextract
from urllib.parse import urlparse
from collections import Counter

SUSPICIOUS_TLDS = {
    "tk","ml","ga","cf","gq","xyz","top","club","online",
    "site","website","space","fun","live","pw","cc","click"
}

BRANDS = [
    "paypal","amazon","google","apple","microsoft","netflix",
    "instagram","facebook","bankofamerica","wellsfargo","chase",
    "linkedin","dropbox","twitter"
]

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    l = len(s)
    return -sum((c/l) * math.log2(c/l) for c in counts.values())

def extract_lexical(url: str) -> dict:
    try:
        parsed  = urlparse(url)
        ext     = tldextract.extract(url)
        domain  = parsed.netloc.lower()
        path    = parsed.path.lower()
        full    = url.lower()
        subdomain = ext.subdomain or ""
        sld       = ext.domain or ""
        tld       = ext.suffix or ""

        return {
            # --- length features ---
            "url_length":         len(full),
            "domain_length":      len(domain),
            "path_length":        len(path),

            # --- character count features ---
            "num_dots":           full.count("."),
            "num_hyphens":        full.count("-"),
            "num_underscores":    full.count("_"),
            "num_slashes":        full.count("/"),
            "num_at":             full.count("@"),
            "num_question":       full.count("?"),
            "num_equals":         full.count("="),
            "num_ampersand":      full.count("&"),
            "num_digits":         sum(c.isdigit() for c in full),
            "digit_ratio":        round(sum(c.isdigit() for c in full)
                                        / max(len(full), 1), 4),

            # --- structural features ---
            "subdomain_count":    len([s for s in subdomain.split(".")
                                       if s]) if subdomain else 0,
            "has_ip":             int(bool(re.match(
                                      r"(\d{1,3}\.){3}\d{1,3}",
                                      domain))),
            "has_port":           int(bool(parsed.port)),
            "uses_https":         int(parsed.scheme == "https"),
            "has_at_symbol":      int("@" in full),
            "double_slash_path":  int("//" in path),
            "has_redirect":       int("url=" in full or
                                      "redirect=" in full),

            # --- entropy ---
            "url_entropy":        round(_entropy(full), 4),
            "domain_entropy":     round(_entropy(sld), 4),

            # --- TLD features ---
            "suspicious_tld":     int(tld in SUSPICIOUS_TLDS),

            # --- brand features ---
            "brand_in_subdomain": int(any(b in subdomain for b in BRANDS)),
            "brand_in_domain":    int(
                any(b in sld and sld != b for b in BRANDS)
            ),

            # --- token features ---
            "num_tokens":         len(re.split(r"[.\-/_=?&]", full)),
            "longest_token":      max(
                (len(t) for t in re.split(r"[.\-/_=?&]", full)),
                default=0
            ),
        }
    except Exception:
        return {}