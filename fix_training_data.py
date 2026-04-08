# fix_training_data.py  — place at project root
import pandas as pd
from urllib.parse import urlparse
import os

CSV_PATH = os.path.join("data", "labelled_domains.csv")

LEGIT_URLS = [
    # E-commerce
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://www.amazon.com/s?k=wireless+headphones",
    "https://www.ebay.com/itm/123456789",
    "https://www.walmart.com/ip/Apple-AirPods/123456",
    "https://www.bestbuy.com/site/apple-airpods/6084400.p",
    "https://www.target.com/p/apple-airpods-pro/-/A-54191097",
    "https://www.etsy.com/listing/123456/handmade-ceramic-mug",
    "https://www.newegg.com/p/N82E16814137733",
    # Developer
    "https://github.com/login",
    "https://github.com/openai/openai-python/issues",
    "https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do",
    "https://docs.python.org/3/library/os.html",
    "https://pypi.org/project/pandas/",
    "https://gitlab.com/users/sign_in",
    "https://hub.docker.com/r/python/python",
    "https://npmjs.com/package/express",
    # Social
    "https://www.linkedin.com/login",
    "https://twitter.com/i/flow/login",
    "https://www.reddit.com/r/MachineLearning/",
    "https://www.facebook.com/login/",
    "https://accounts.google.com/signin/v2/identifier",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://outlook.live.com/mail/0/inbox",
    # Finance
    "https://www.chase.com/personal/banking/login",
    "https://www.bankofamerica.com/online-banking/sign-in/",
    "https://login.microsoftonline.com/common/oauth2/authorize",
    "https://www.coinbase.com/signin",
    "https://app.robinhood.com/login/",
    # Travel
    "https://www.booking.com/searchresults.html?ss=Paris",
    "https://www.airbnb.com/s/London/homes",
    "https://www.expedia.com/Hotel-Search?destination=New+York",
    "https://www.tripadvisor.com/Hotels-g60763-New_York_City.html",
    # News
    "https://www.bbc.com/news/world-us-canada",
    "https://www.nytimes.com/section/technology",
    "https://www.reuters.com/technology/",
    "https://www.cnn.com/us",
    # Cloud / SaaS
    "https://console.aws.amazon.com/ec2/v2/home",
    "https://portal.azure.com/#blade/Microsoft_AAD_IAM/ActiveDirectoryMenuBlade",
    "https://app.netlify.com/sites",
    "https://dashboard.heroku.com/apps",
    "https://vercel.com/dashboard",
    # Education
    "https://www.coursera.org/learn/machine-learning",
    "https://www.udemy.com/course/the-complete-python-bootcamp/",
    "https://www.khanacademy.org/math/linear-algebra",
    "https://www.wikipedia.org/wiki/Phishing",
    "https://arxiv.org/abs/2305.15047",
    # Health
    "https://www.webmd.com/cold-and-flu/flu-guide/what-is-flu",
    "https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444",
    "https://www.nih.gov/health-information",
]

def main():
    df = pd.read_csv(CSV_PATH)
    print(f"[before] rows: {len(df)}")
    print(df["label"].value_counts().to_string())

    new_rows = []
    for url in LEGIT_URLS:
        domain = urlparse(url).netloc
        # Build a row matching existing columns; unknown cols stay None
        row = {col: None for col in df.columns}
        row["url"]    = url
        row["domain"] = domain
        row["label"]  = 0
        row["source"] = "manual_legit_paths"
        new_rows.append(row)

    combined = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_csv(CSV_PATH, index=False)

    print(f"\n[after]  rows: {len(combined)}")
    print(combined["label"].value_counts().to_string())
    print(f"\nDone — added {len(new_rows)} legit URLs with paths.")

if __name__ == "__main__":
    main()