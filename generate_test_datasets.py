import pandas as pd
import numpy as np

print("Loading original dataset...")
df_orig = pd.read_csv('loan_approval_10000.csv')
num_rows = 2000  # Generate 2000 rows for each test file

# Helper to get random sample
def get_sample(n):
    return df_orig.sample(n=n, replace=True).copy().reset_index(drop=True)

# 1. GOOD / STABLE DATASET (No Drift)
print("Generating stable (no drift) dataset...")
df_stable = get_sample(num_rows)
# Add minor normal random noise (less than 2%) to numeric columns to make it feel fresh but statistically identical
df_stable['income_annum'] = (df_stable['income_annum'] * np.random.uniform(0.99, 1.01, num_rows)).astype(int)
df_stable['loan_amount'] = (df_stable['loan_amount'] * np.random.uniform(0.99, 1.01, num_rows)).astype(int)
df_stable['cibil_score'] = (df_stable['cibil_score'] + np.random.randint(-5, 6, num_rows)).clip(300, 900)
df_stable.to_csv('loan_approval_stable.csv', index=False)
print("Saved: loan_approval_stable.csv (Good / No Drift)")

# 2. MEDIUM / MODERATE DRIFT DATASET
print("Generating moderate drift dataset...")
df_mod = get_sample(num_rows)
# Drift 2 features moderately:
# - Shift CIBIL score average down by 25 points
df_mod['cibil_score'] = (df_mod['cibil_score'] - 25).clip(lower=300)
# - Shift loan_term up by 2 years on average
df_mod['loan_term'] = (df_mod['loan_term'] + 2).clip(2, 20)
# - Add minor noise to others
df_mod['income_annum'] = (df_mod['income_annum'] * np.random.uniform(0.98, 1.02, num_rows)).astype(int)
df_mod.to_csv('loan_approval_moderate_drift.csv', index=False)
print("Saved: loan_approval_moderate_drift.csv (Medium Drift)")

# 3. BAD / HIGH DRIFT DATASET
print("Generating high drift dataset...")
df_high = get_sample(num_rows)
# Drift 4 features significantly:
# - CIBIL score average down by 75 points
df_high['cibil_score'] = (df_high['cibil_score'] - 75).clip(lower=300)
# - Increase loan amounts by 1.8x
df_high['loan_amount'] = (df_high['loan_amount'] * 1.8).astype(int)
# - Increase annual income by 1.5x
df_high['income_annum'] = (df_high['income_annum'] * 1.5).astype(int)
# - Shift dependents count (mostly 4 or 5 dependents)
df_high['no_of_dependents'] = np.random.choice([4, 5], size=num_rows, p=[0.4, 0.6])
df_high.to_csv('loan_approval_high_drift.csv', index=False)
print("Saved: loan_approval_high_drift.csv (Bad / High Drift)")

# 4. WORST / EXTREME DRIFT DATASET
print("Generating extreme drift dataset...")
df_worst = get_sample(num_rows)
# Drift almost all features drastically:
# - CIBIL score crashed down by 180 points (huge risk shift)
df_worst['cibil_score'] = (df_worst['cibil_score'] - 180).clip(lower=300)
# - Loan amounts spiked by 3.2x
df_worst['loan_amount'] = (df_worst['loan_amount'] * 3.2).astype(int)
# - Annual income dropped by 30%
df_worst['income_annum'] = (df_worst['income_annum'] * 0.7).astype(int)
# - Asset values plummeted by 50%
df_worst['residential_assets_value'] = (df_worst['residential_assets_value'] * 0.5).astype(int)
df_worst['commercial_assets_value'] = (df_worst['commercial_assets_value'] * 0.5).astype(int)
df_worst['bank_asset_value'] = (df_worst['bank_asset_value'] * 0.5).astype(int)
# - Loan term set to maximum (20 years) for everyone
df_worst['loan_term'] = 20
# - Make everyone self-employed (higher risk profile)
df_worst['self_employed'] = 'Yes'
df_worst.to_csv('loan_approval_extreme_drift.csv', index=False)
print("Saved: loan_approval_extreme_drift.csv (Worst / Extreme Drift)")

print("\nAll 4 test datasets successfully generated in workspace!")
