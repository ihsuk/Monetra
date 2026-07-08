import pandas as pd
import numpy as np

print("Loading original dataset...")
df = pd.read_csv('loan_approval_10000.csv')

# Create a massive drift by shifting distributions
# 1. Multiply loan amounts by 3x (massive shift)
df['loan_amount'] = df['loan_amount'] * 3.5

# 2. Add 20 years to applicant age
if 'applicant_age' in df.columns:
    df['applicant_age'] = df['applicant_age'] + 20
else:
    df['applicant_age'] = np.random.randint(40, 70, size=len(df))

# 3. Drop CIBIL scores by 150 points
df['cibil_score'] = (df['cibil_score'] - 150).clip(lower=300)

# 4. Increase income massively
df['income_annum'] = df['income_annum'] * 2.5

# 5. Make everyone have 20 year loan terms
df['loan_term'] = 20

# Save to a new CSV file specifically for the drift demo
output_file = 'loan_approval_high_drift_demo.csv'
df.to_csv(output_file, index=False)
print(f"Success! A new drifted dataset was saved to: {output_file}")
print("You can upload this file in the dashboard to demonstrate large data drift.")
