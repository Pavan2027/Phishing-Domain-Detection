# features/pipeline.py
import pandas as pd
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from features.lexical       import extract_lexical
from features.rdap_ssl      import extract_rdap_ssl
from features.html_features import extract_html

# Suppress BeautifulSoup XML-parsed-as-HTML spam
from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Per-domain wall-clock timeout (seconds).
# RDAP has its own 6s timeout internally; this is a safety net
# for the entire row (lexical + RDAP + HTML combined).
ROW_TIMEOUT = 12


def _extract_row(row: dict) -> dict:
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
    """Return set of URLs already processed in a previous run."""
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
    """Append a completed batch to the checkpoint CSV."""
    if not batch:
        return
    df = pd.DataFrame(batch)
    write_header = not os.path.exists(checkpoint_path)
    df.to_csv(checkpoint_path, mode="a", index=False, header=write_header)


def build_feature_matrix(
    csv_path:        str = "data/labelled_domains.csv",
    out_path:        str = "data/feature_matrix.csv",
    checkpoint_path: str = "data/feature_matrix_checkpoint.csv",
    max_workers:     int = 500,   # 500 is the sweet spot on Windows
    batch_size:      int = 2000,  # flush to disk every N completions
    limit:           int = None
):
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    # ── Resume: skip rows already in checkpoint ──────────────────────────
    done_urls = _load_checkpoint(checkpoint_path)
    remaining = df[~df["url"].astype(str).isin(done_urls)]
    skipped   = len(df) - len(remaining)
    total_all = len(df)

    print(f"Total rows   : {total_all:,}")
    print(f"Already done : {skipped:,}")
    print(f"To process   : {len(remaining):,}")
    print(f"Workers      : {max_workers}  |  Batch size: {batch_size}")
    print(f"Row timeout  : {ROW_TIMEOUT}s per domain\n")

    if remaining.empty:
        print("Nothing left to process — merging checkpoint into final output.")
        _finalise(checkpoint_path, out_path, total_all)
        return pd.read_csv(out_path)

    rows = remaining.to_dict("records")

    completed  = skipped        # total done including prior runs
    errors     = 0
    batch      = []             # in-memory buffer before each flush
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_row, row): row
            for row in rows
        }

        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result(timeout=ROW_TIMEOUT)
                batch.append(result)
            except TimeoutError:
                # Row took too long — keep lexical features at minimum
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

            # ── Flush batch to checkpoint ────────────────────────────────
            if len(batch) >= batch_size:
                _append_checkpoint(batch, checkpoint_path)
                batch = []

            # ── Progress line ────────────────────────────────────────────
            if completed % 1000 == 0 or completed == total_all:
                elapsed  = time.time() - start_time
                rate     = (completed - skipped) / max(elapsed, 1)
                remaining_count = total_all - completed
                eta_sec  = remaining_count / max(rate, 1)
                eta_min  = eta_sec / 60
                pct      = completed / total_all * 100
                bar_fill = int(pct / 5)
                bar      = "█" * bar_fill + "░" * (20 - bar_fill)
                print(
                    f"  [{bar}] {pct:5.1f}%  "
                    f"{completed:,}/{total_all:,}  "
                    f"errors: {errors}  "
                    f"rate: {rate:.0f}/s  "
                    f"ETA: {eta_min:.1f}m"
                )

    # ── Flush remaining batch ────────────────────────────────────────────
    if batch:
        _append_checkpoint(batch, checkpoint_path)

    # ── Merge checkpoint → final output ─────────────────────────────────
    _finalise(checkpoint_path, out_path, total_all)
    return pd.read_csv(out_path)


def _finalise(checkpoint_path: str, out_path: str, expected_rows: int):
    """Deduplicate checkpoint and write final feature_matrix.csv."""
    print(f"\nFinalising → {out_path}")
    df = pd.read_csv(checkpoint_path)

    # Drop dupes (can appear if a prior run was interrupted mid-flush)
    before = len(df)
    df = df.drop_duplicates(subset=["url"])
    if before != len(df):
        print(f"  Removed {before - len(df)} duplicate rows from checkpoint")

    df.to_csv(out_path, index=False)
    print(f"  Shape    : {df.shape}")
    print(f"  Expected : {expected_rows:,} rows")
    if len(df) < expected_rows:
        print(f"  WARNING  : {expected_rows - len(df)} rows missing — "
              f"re-run to fill gaps (checkpoint will be reused)")
    else:
        print(f"  Feature matrix complete.")
        # Clean up checkpoint only when fully done
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"  Checkpoint removed.")


if __name__ == "__main__":
    build_feature_matrix()