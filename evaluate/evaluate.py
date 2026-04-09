# evaluate/run_phase5.py
from confusion_report  import full_report
from threshold_tuning  import tune_threshold
from shap_analysis     import run_shap
from lime_explain      import explain_url

print("\n" + "="*50)
print("STEP 1/4 — Confusion matrix + ROC curve")
print("="*50)
full_report(threshold=0.5)

print("\n" + "="*50)
print("STEP 2/4 — Threshold tuning")
print("="*50)
best_t = tune_threshold(min_precision=0.90)

print("\n" + "="*50)
print("STEP 3/4 — Re-running report at tuned threshold")
print("="*50)
full_report(threshold=best_t)

print("\n" + "="*50)
print("STEP 4/4 — SHAP feature importance")
print("="*50)
run_shap(sample_n=500)

print("\n" + "="*50)
print("DONE — all outputs in evaluate/outputs/")
print("Run lime_explain.py separately per URL.")
print("="*50)