# stress_test.py
import joblib
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from features.lexical import extract_lexical

# ─────────────────────────────────────────────────────────────
# Load trained model assets
# ─────────────────────────────────────────────────────────────
model    = joblib.load("models/best_model.pkl")
features = joblib.load("models/selected_features.pkl")
scaler   = joblib.load("models/scaler.pkl")

# ─────────────────────────────────────────────────────────────
# Stress Test Cases
# ─────────────────────────────────────────────────────────────
TEST_URLS = [

    # ── CLEAR PHISHING ────────────────────────────────────────
    ("http://paypal-secure-login.xyz/verify/account?id=1234", "PHISHING"),
    ("http://amazon-account-update.tk/login.php", "PHISHING"),
    ("http://192.168.1.1/bank/login", "PHISHING"),
    ("http://microsoft-support-alert.cf/fix?session=abc123", "PHISHING"),
    ("http://paypal.com.evil-site.xyz/signin", "PHISHING"),
    ("http://secure.apple.com.phish.tk/verify", "PHISHING"),
    ("http://google.accounts.login-verify.top/auth", "PHISHING"),
    ("http://amazon.customer-service.ml/help", "PHISHING"),
    ("http://netflix-login.club/account/verify", "PHISHING"),
    ("http://chase-banking.pw/secure/login", "PHISHING"),
    ("http://dropbox-shared.space/download?file=invoice.exe", "PHISHING"),
    ("http://xk29fmq7.xyz/login", "PHISHING"),
    ("http://a8f3k2p9.top/verify/account", "PHISHING"),
    ("http://45.33.32.156/admin/login.php", "PHISHING"),
    ("http://185.220.101.47/bank/verify?token=abc", "PHISHING"),
    ("http://evil.xyz/redirect?url=https://paypal.com", "PHISHING"),
    ("http://phish.tk/go?url=https://google.com&id=steal", "PHISHING"),
    ("http://wellsfargo-secure.cf/online/signin/index.html", "PHISHING"),
    ("http://update-billing-info.top/netflix/payment", "PHISHING"),

    # ── TRICKY PHISHING ───────────────────────────────────────
    ("http://login.ml", "PHISHING"),
    ("http://secure.tk", "PHISHING"),
    ("https://signin.pw/account", "PHISHING"),
    ("https://paypal-account-verify.xyz/secure/login", "PHISHING"),
    ("https://apple-id-locked.top/unlock", "PHISHING"),
    ("https://microsoft365-login.club/auth", "PHISHING"),
    ("http://paypa1.com/signin", "PHISHING"),
    ("http://arnazon.com/login", "PHISHING"),
    ("http://g00gle.com/accounts", "PHISHING"),
    ("http://micosoft.com/login", "PHISHING"),
    ("http://secure-payment.xyz/paypal/com/myaccount/login", "PHISHING"),
    ("http://account-verify.top/google/com/signin/v2", "PHISHING"),
    ("http://phish.xyz/login%20page%3Fid%3D1234", "PHISHING"),

    # ── CLEAR LEGIT ───────────────────────────────────────────
    ("https://www.google.com/search?q=phishing+detection", "legit"),
    ("https://accounts.google.com/signin/v2/identifier", "legit"),
    ("https://mail.google.com/mail/u/0/#inbox", "legit"),
    ("https://www.amazon.com/dp/B08N5WRWNW", "legit"),
    ("https://www.amazon.com/s?k=laptop&ref=nb_sb_noss", "legit"),
    ("https://github.com/login", "legit"),
    ("https://github.com/openai/openai-python/issues/123", "legit"),
    ("https://www.microsoft.com/en-us/microsoft-365", "legit"),
    ("https://login.microsoftonline.com/common/oauth2/authorize", "legit"),
    ("https://www.amazon.com/gp/product/B08N5WRWNW/ref=ppx_yo_dt_b_asin_title_o01_s00?ie=UTF8&psc=1", "legit"),
    ("https://www.google.com/search?q=machine+learning&rlz=1C1GCEU&oq=machine&aqs=chrome", "legit"),
    ("https://accounts.google.com/signin/v2/identifier?hl=en&passive=true&continue=https%3A%2F%2Fmail.google.com", "legit"),
    ("https://mail.google.com/mail/u/0/", "legit"),
    ("https://console.aws.amazon.com/ec2/v2/home", "legit"),
    ("https://portal.azure.com/#home", "legit"),
    ("https://app.slack.com/signin", "legit"),
    ("https://support.apple.com/en-us/HT201272", "legit"),
    ("https://developer.mozilla.org/en-US/docs/Web/API", "legit"),
    ("https://www.chase.com/personal/banking/login", "legit"),
    ("https://www.bankofamerica.com/online-banking/sign-in/", "legit"),
    ("https://www.paypal.com/signin", "legit"),
    ("https://www.netflix.com/login", "legit"),
    ("https://appleid.apple.com/sign-in", "legit"),
    ("https://www.dropbox.com/login", "legit"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "legit"),
    ("https://stackoverflow.com/questions/231767/yield-keyword", "legit"),
    ("https://arxiv.org/abs/2305.15047", "legit"),

    # ── EDGE CASES ────────────────────────────────────────────
    ("https://www.paypal.com/myaccount/summary", "legit"),
    ("https://paypal.com.secure-login.xyz/myaccount/summary", "PHISHING"),
    ("https://www.google.com/landing/2step/", "legit"),
    ("https://myaccount.google.com/security", "legit"),
    ("https://www.amazon.com/gp/css/account/info/view.html", "legit"),
    ("https://bit.ly/3xYzAbc", "legit"),
    ("https://t.co/AbCdEfGhIj", "legit"),
    ("https://tinyurl.com/y4jxm9pz", "legit"),
    ("https://www.irs.gov/filing", "legit"),
    ("https://www.ssa.gov/benefits/", "legit"),
    ("https://ocw.mit.edu/courses/", "legit"),
    ("https://arxiv.org/abs/1706.03762", "legit"),
    ("https://www.netflix.com/browse", "legit"),
    ("https://www.instagram.com/accounts/login/", "legit"),
    ("https://www.linkedin.com/feed/", "legit"),
]

# ─────────────────────────────────────────────────────────────
# Run Stress Test
# ─────────────────────────────────────────────────────────────
correct = 0
wrong = []

print(f"{'Result':<6} {'Expected':<10} {'Conf':>7}  URL")
print("-" * 100)

for url, expected in TEST_URLS:
    feats = extract_lexical(url)

    row = pd.DataFrame([feats])
    for f in features:
        if f not in row.columns:
            row[f] = -1

    row = row[features].fillna(-1)
    prob = model.predict_proba(scaler.transform(row))[0][1]

    pred = "PHISHING" if prob >= 0.5 else "legit"
    ok = pred == expected

    marker = "✓" if ok else "✗"
    correct += int(ok)

    if not ok:
        wrong.append((url, expected, pred, prob))

    display = url[:70] + "..." if len(url) > 73 else url
    print(f"{marker:<6} {expected:<10} {prob:>6.1%}  {display}")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"Score: {correct}/{len(TEST_URLS)} ({correct/len(TEST_URLS)*100:.1f}%)")

if wrong:
    print(f"\nFailed Cases ({len(wrong)}):")
    for url, exp, pred, prob in wrong:
        print(f"Expected {exp:<10} got {pred:<10} ({prob:.1%}) -> {url}")
else:
    print("\nAll test cases passed successfully.")