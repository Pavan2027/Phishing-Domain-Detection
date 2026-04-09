# labelling/export_labelled.py
import sqlite3, pandas as pd

def export_labelled():
    conn = sqlite3.connect("data/phishing.db")
    df   = pd.read_sql("""
        SELECT url, domain, label, source,
               domain_age_days, registrar,
               country, scraped_at
        FROM domains
        WHERE label IN (0, 1)
    """, conn)
    conn.close()

    before = len(df)

    # Deduplicate by URL — NOT by domain.
    # Deduping by domain was dropping augmented legit URLs that share a
    # domain with the original bare Tranco entry.
    df = df.drop_duplicates(subset=["url"])
    print(f"Rows after dedup: {len(df):,} (removed {before - len(df):,} dupes)")

    df.to_csv("data/labelled_domains.csv", index=False)
    print(f"\nExported → data/labelled_domains.csv")
    print(f"Shape: {df.shape}")
    print("\nLabel breakdown:")
    print(df["label"].value_counts().to_string())
    print("\nSource breakdown:")
    print(df["source"].value_counts().to_string())

if __name__ == "__main__":
    export_labelled()