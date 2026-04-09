# fix_training_data.py  — place at project root
#
# Generates exactly 1 variant per Tranco domain.
# Previous run used 4 variants → 124k legit vs 25k phishing (4.85:1)
# → SMOTE generated 74k synthetic phishing → poisoned decision boundary.
#
# 1 variant → ~50k legit vs 25k phishing (2:1) → no SMOTE needed.
# Each variant alternates between www and no-www so the model sees both.

import sqlite3
import pandas as pd
from datetime import datetime
import os
import random

DB_PATH = os.path.join("data", "phishing.db")

# Representative paths — balanced between short and deep
PATHS = [
    "/", "/login", "/signin", "/sign-in", "/register", "/signup",
    "/account", "/account/settings", "/profile", "/dashboard",
    "/search?q=test", "/help", "/support", "/privacy", "/terms",
    "/products", "/services", "/blog", "/news", "/pricing",
    "/download", "/docs", "/documentation", "/en/home", "/cart",
    "/checkout", "/orders", "/shop", "/feed", "/notifications",
    "/messages", "/settings/security", "/verify", "/reset-password",
    "/oauth/authorize", "/auth/login", "/user/profile", "/users/sign_in",
    "/watch?v=dQw4w9WgXcQ", "/article/breaking-news-today",
    "/s?k=search+term", "/search?q=example&page=1",
    "/gp/sign-in.html", "/dp/B08N5WRWNW", "/itm/123456789",
]


def make_variants(bare_domain: str, index: int) -> list:
    path = PATHS[index % len(PATHS)]
    results = []

    # variant A — with a path (existing behaviour)
    if index % 2 == 0:
        full_domain = f"www.{bare_domain}"
    else:
        full_domain = bare_domain
    results.append((f"https://{full_domain}{path}", full_domain))

    # variant B — bare www, no path (fills the google.com gap)
    www_domain = f"www.{bare_domain}"
    results.append((f"https://{www_domain}", www_domain))

    return results


def main():
    random.seed(42)
    conn = sqlite3.connect(DB_PATH)

    # ── Step 1: Fix Tranco http:// → https:// ───────────────────────────
    cur = conn.execute(
        "SELECT COUNT(*) FROM domains "
        "WHERE label=0 AND source='tranco' AND url LIKE 'http://%'"
    )
    http_count = cur.fetchone()[0]
    if http_count > 0:
        conn.execute("""
            UPDATE domains
            SET url = 'https://' || SUBSTR(url, 8)
            WHERE label=0 AND source='tranco' AND url LIKE 'http://%'
        """)
        conn.commit()
        print(f"[Step 1] Upgraded {http_count:,} Tranco URLs: http:// → https://")
    else:
        print("[Step 1] Tranco URLs already use https://")

    # ── Step 2: Remove ALL previously augmented rows ─────────────────────
    cur = conn.execute(
        "SELECT COUNT(*) FROM domains WHERE source='tranco_augmented'"
    )
    old_count = cur.fetchone()[0]
    if old_count > 0:
        conn.execute("DELETE FROM domains WHERE source='tranco_augmented'")
        conn.commit()
        print(f"[Step 2] Removed {old_count:,} old augmented rows")
    else:
        print("[Step 2] No old augmented rows found")

    # ── Step 3: Load Tranco domains ──────────────────────────────────────
    tranco_df = pd.read_sql(
        "SELECT url, domain FROM domains WHERE label=0 AND source='tranco'",
        conn
    )
    print(f"[Step 3] Loaded {len(tranco_df):,} Tranco domains")

    # ── Step 4: Load existing URLs to avoid dupes ────────────────────────
    existing = pd.read_sql("SELECT url FROM domains", conn)
    existing_urls = set(existing["url"].astype(str).tolist())

    # ── Step 5: Insert 1 variant per domain ──────────────────────────────
    now      = datetime.utcnow().isoformat()
    inserted = 0
    skipped  = 0
    batch    = []
    BATCH_SIZE = 2000

    for i, row in enumerate(tranco_df.itertuples()):
        bare = row.domain if row.domain else \
            str(row.url).replace("https://", "").replace("http://", "").strip("/")

        for url, full_domain in make_variants(bare, i):
            if url in existing_urls:
                skipped += 1
                continue

            batch.append((url, full_domain, 0, "tranco_augmented", 1, now))
            existing_urls.add(url)
            inserted += 1

            if len(batch) >= BATCH_SIZE:
                conn.executemany(
                    "INSERT INTO domains (url, domain, label, source, verified, scraped_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    batch
                )
                conn.commit()
                batch = []

    if batch:
        conn.executemany(
            "INSERT INTO domains (url, domain, label, source, verified, scraped_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            batch
        )
        conn.commit()

    print(f"[Step 4] Inserted {inserted:,} variants ({skipped} skipped as dupes)")

    # ── Step 5: Verify final counts ──────────────────────────────────────
    counts = pd.read_sql(
        "SELECT label, COUNT(*) as n FROM domains WHERE label IN (0,1) GROUP BY label",
        conn
    )
    total_legit   = counts[counts["label"] == 0]["n"].values[0]
    total_phish   = counts[counts["label"] == 1]["n"].values[0]
    ratio         = total_legit / total_phish

    print(f"\n[Step 5] Final DB counts:")
    print(f"  Legit    : {total_legit:,}")
    print(f"  Phishing : {total_phish:,}")
    print(f"  Ratio    : {ratio:.2f}:1  ({'SMOTE will NOT trigger' if ratio <= 4.0 else 'SMOTE may trigger'})")

    sample = pd.read_sql(
        "SELECT url FROM domains WHERE source='tranco_augmented' LIMIT 8", conn
    )
    print(f"\n  Sample augmented URLs:")
    for url in sample["url"].tolist():
        print(f"    {url}")

    conn.close()
    print(f"\nDone. Now run:")
    print("  python labelling/export_labelled.py")
    print("  python -m features.pipeline")
    print("  python -m features.preprocess")
    print("  python training/train_all.py")
    print("  python test_model.py")


if __name__ == "__main__":
    main()