# features/pipeline.py
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError
from features.lexical import extract_lexical

# Suppress BeautifulSoup XML-parsed-as-HTML spam
from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ROW_TIMEOUT = 12


def _extract_row_lexical_only(row: dict) -> dict:
    """Lexical features only — no network calls. Fast, CPU-bound."""
    url    = str(row["url"])
    domain = str(row["domain"])
    feats  = {
        "url":    url,
        "domain": domain,
        "label":  row["label"],
    }
    feats.update(extract_lexical(url))
    return feats


def _extract_row_full(row: dict) -> dict:
    """All features including RDAP, SSL, HTML — slow, network-bound."""
    from features.rdap_ssl      import extract_rdap_ssl
    from features.html_features import extract_html
    url    = str(row["url"])
    domain = str(row["domain"])
    feats  = {
        "url":    url,
        "domain": domain,
        "label":  row["label"],
    }
    feats.update(extract_lexical(url))
    feats.update(extract_rdap_ssl(domain))
    feats.update(extract_html(url))
    return feats


def _load_checkpoint(checkpoint_path: str) -> set:
    if not os.path.exists(checkpoint_path):
        return set()
    try:
        done = pd.read_csv(checkpoint_path, usecols=["url"])
        urls = set(done["url"].astype(str).tolist())
        print(f"[resume] checkpoint found — {len(urls):,} rows already done, skipping them")
        return urls
    except Exception:
        return set()


def _append_checkpoint(batch: list, checkpoint_path: str):
    if not batch:
        return
    df = pd.DataFrame(batch)
    write_header = not os.path.exists(checkpoint_path)
    df.to_csv(checkpoint_path, mode="a", index=False, header=write_header)


def build_feature_matrix(
    csv_path:        str  = "data/labelled_domains.csv",
    out_path:        str  = "data/feature_matrix.csv",
    checkpoint_path: str  = "data/feature_matrix_checkpoint.csv",
    lexical_only:    bool = True,   # ← True = skip network calls entirely
                                    #   False = run full pipeline (RDAP/SSL/HTML)
                                    #   Since preprocess.py uses LEXICAL_ONLY,
                                    #   there is no benefit to setting this False
                                    #   unless you plan to use network features.
    max_workers:     int  = None,   # None = auto (see below)
    batch_size:      int  = 5000,
    limit:           int  = None,
):
    df = pd.read_csv(csv_path, low_memory=False, dtype={"registrar": str, "country": str})
    if limit:
        df = df.head(limit)

    # ── Auto worker count ────────────────────────────────────────────────
    # Lexical extraction is pure CPU — use ProcessPoolExecutor with one
    # worker per logical core for true parallelism (bypasses Python GIL).
    # Network extraction is I/O bound — ThreadPoolExecutor, 500 workers.
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()

    if max_workers is None:
        if lexical_only:
            # CPU-bound: one process per logical core
            max_workers = cpu_count
        else:
            # I/O-bound: threads, Windows socket limit ~500
            max_workers = 500

    mode_label = "lexical-only (CPU, ProcessPool)" if lexical_only \
                 else "full network (I/O, ThreadPool)"

    # ── Resume from checkpoint ───────────────────────────────────────────
    done_urls = _load_checkpoint(checkpoint_path)
    remaining = df[~df["url"].astype(str).isin(done_urls)]
    skipped   = len(df) - len(remaining)
    total_all = len(df)

    print(f"Mode         : {mode_label}")
    print(f"Workers      : {max_workers}  ({'processes' if lexical_only else 'threads'})")
    print(f"Total rows   : {total_all:,}")
    print(f"Already done : {skipped:,}")
    print(f"To process   : {len(remaining):,}")
    print(f"Batch size   : {batch_size:,}\n")

    if remaining.empty:
        print("Nothing left to process — merging checkpoint into final output.")
        _finalise(checkpoint_path, out_path, total_all)
        return pd.read_csv(out_path)

    rows      = remaining.to_dict("records")
    completed = skipped
    errors    = 0
    batch     = []
    start_time = time.time()

    extract_fn = _extract_row_lexical_only if lexical_only else _extract_row_full
    Executor   = ProcessPoolExecutor if lexical_only else ThreadPoolExecutor

    with Executor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_fn, row): row for row in rows}

        for future in as_completed(futures):
            row = futures[future]
            try:
                if lexical_only:
                    result = future.result()
                else:
                    result = future.result(timeout=ROW_TIMEOUT)
                batch.append(result)
            except TimeoutError:
                errors += 1
                batch.append({
                    "url":    str(row["url"]),
                    "domain": str(row["domain"]),
                    "label":  row["label"],
                    **extract_lexical(str(row["url"]))
                })
            except Exception:
                errors += 1
                batch.append({
                    "url":    str(row["url"]),
                    "domain": str(row["domain"]),
                    "label":  row["label"],
                })

            completed += 1

            if len(batch) >= batch_size:
                _append_checkpoint(batch, checkpoint_path)
                batch = []

            if completed % 5000 == 0 or completed == total_all:
                elapsed         = time.time() - start_time
                rate            = (completed - skipped) / max(elapsed, 1)
                remaining_count = total_all - completed
                eta_min         = (remaining_count / max(rate, 1)) / 60
                pct             = completed / total_all * 100
                bar             = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(
                    f"  [{bar}] {pct:5.1f}%  "
                    f"{completed:,}/{total_all:,}  "
                    f"errors: {errors}  "
                    f"rate: {rate:.0f}/s  "
                    f"ETA: {eta_min:.1f}m"
                )

    if batch:
        _append_checkpoint(batch, checkpoint_path)

    _finalise(checkpoint_path, out_path, total_all)
    return pd.read_csv(out_path)


def _finalise(checkpoint_path: str, out_path: str, expected_rows: int):
    print(f"\nFinalising → {out_path}")
    df = pd.read_csv(checkpoint_path)

    before = len(df)
    df = df.drop_duplicates(subset=["url"])
    if before != len(df):
        print(f"  Removed {before - len(df)} duplicate rows from checkpoint")

    df.to_csv(out_path, index=False)
    print(f"  Shape    : {df.shape}")
    print(f"  Expected : {expected_rows:,} rows")
    if len(df) < expected_rows:
        print(f"  WARNING  : {expected_rows - len(df)} rows missing — re-run to fill gaps")
    else:
        print(f"  Feature matrix complete.")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"  Checkpoint removed.")


if __name__ == "__main__":
    build_feature_matrix()