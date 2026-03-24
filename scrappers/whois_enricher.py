# scrapers/whois_enricher.py
import whois
import pandas as pd
import sqlite3
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── how many parallel threads to use ─────────────────────────────────────────
# 50 is the sweet spot — beyond this WHOIS servers start refusing connections
MAX_WORKERS  = 50
BATCH_SIZE   = 500    # commit to DB every N domains so you don't lose progress
RETRY_LIMIT  = 2      # retry a domain this many times before giving up


def _query_single(domain: str, attempt: int = 0) -> dict:
    """Query WHOIS for one domain. Returns a flat dict of features."""
    base = {
        "domain":          domain,
        "registrar":       None,
        "creation_date":   None,
        "country":         None,
        "domain_age_days": None,
        "privacy_protected": 0,
        "whois_error":     None,
        "fetched_at":      datetime.utcnow().isoformat(),
    }
    try:
        w = whois.whois(domain)

        created = w.creation_date
        if isinstance(created, list):
            created = created[0]   # some registrars return a list

        registrar = str(w.registrar or "").lower()

        base.update({
            "registrar":       registrar[:80],
            "creation_date":   str(created) if created else None,
            "country":         str(w.country or "").upper(),
            "domain_age_days": (datetime.utcnow() - created).days
                               if created else None,
            "privacy_protected": int(any(k in registrar for k in
                                   ["privacy","proxy","protect","guard","whois"])),
        })

    except whois.parser.PywhoisError as e:
        # domain genuinely has no WHOIS record (common for new/phishing domains)
        base["whois_error"] = "no_record"

    except Exception as e:
        base["whois_error"] = str(e)[:120]
        # retry once on transient errors (timeout, connection reset)
        if attempt < RETRY_LIMIT:
            time.sleep(1)
            return _query_single(domain, attempt + 1)

    return base


def enrich_with_whois_parallel(domains: list,
                                db_path: str = "data/phishing.db") -> pd.DataFrame:
    """
    Enrich a list of domains with WHOIS data using a thread pool.
    Saves results to the whois_cache table in batches so progress
    is never lost if the run crashes halfway through.
    """
    # ── skip domains already in cache ─────────────────────────────────────────
    conn   = sqlite3.connect(db_path)
    cached = pd.read_sql("SELECT domain FROM whois_cache", conn)
    conn.close()
    already_done = set(cached["domain"].tolist())
    todo = [d for d in domains if d not in already_done]

    log.info(f"Total domains: {len(domains)} | "
             f"Cached: {len(already_done)} | To fetch: {len(todo)}")

    if not todo:
        log.info("All domains already cached.")
        return _load_cache(db_path)

    results = []
    errors  = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_query_single, d): d for d in todo}

        for i, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
                results.append(row)
            except Exception as e:
                domain = futures[future]
                log.warning(f"Unhandled error for {domain}: {e}")
                errors += 1

            # ── progress logging ───────────────────────────────────────────
            if i % 100 == 0:
                pct = (i / len(todo)) * 100
                log.info(f"  {i}/{len(todo)} ({pct:.1f}%) | errors so far: {errors}")

            # ── batch commit to DB so crashes don't lose progress ──────────
            if len(results) >= BATCH_SIZE:
                _save_batch(results, db_path)
                results = []

    # save any remaining results
    if results:
        _save_batch(results, db_path)

    log.info(f"Done. Total errors: {errors}")
    return _load_cache(db_path)


def _save_batch(rows: list, db_path: str):
    """Insert a batch into whois_cache, ignoring duplicates."""
    df   = pd.DataFrame(rows)
    conn = sqlite3.connect(db_path)
    df.to_sql("whois_cache", conn, if_exists="append",
              index=False, method="multi")
    conn.execute("""
        DELETE FROM whois_cache
        WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM whois_cache GROUP BY domain
        )
    """)
    conn.commit()
    conn.close()
    log.info(f"  Committed batch of {len(rows)} to whois_cache")


def _load_cache(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("SELECT * FROM whois_cache", conn)
    conn.close()
    return df