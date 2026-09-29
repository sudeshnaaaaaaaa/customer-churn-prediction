"""
generate_data.py
Creates a synthetic telecom customer churn dataset with realistic,
correlated patterns (not random noise) so the ML pipeline has genuine
signal to learn from — mirroring the structure of the well-known
Telco Customer Churn dataset.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 2000

tenure = rng.integers(0, 73, n)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
monthly_charges = np.round(rng.normal(65, 25, n).clip(18, 120), 2)
tech_support = rng.choice(["Yes", "No"], n, p=[0.4, 0.6])
online_security = rng.choice(["Yes", "No"], n, p=[0.4, 0.6])
paperless_billing = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])
payment_method = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
)
senior_citizen = rng.choice([0, 1], n, p=[0.84, 0.16])
partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
num_support_calls = rng.poisson(1.5, n)

total_charges = np.round(monthly_charges * tenure + rng.normal(0, 50, n).clip(-100, 100), 2)
total_charges = total_charges.clip(0, None)

# Build churn probability from realistic drivers (this is the actual "signal"
# the model has to learn — short tenure, month-to-month, fiber+no support,
# high charges, and frequent support calls all raise churn risk)
logit = (
    -2.0
    + (contract == "Month-to-month") * 1.1
    + (contract == "One year") * 0.1
    - (tenure / 72) * 1.8
    + (internet_service == "Fiber optic") * 0.5
    + (tech_support == "No") * 0.5
    + (online_security == "No") * 0.3
    + (monthly_charges - 65) / 40 * 0.6
    + (num_support_calls >= 3) * 0.9
    + (payment_method == "Electronic check") * 0.4
    + rng.normal(0, 0.6, n)
)
prob = 1 / (1 + np.exp(-logit))
churn = (rng.random(n) < prob).astype(int)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame({
    "customer_id": [f"CUST{i:05d}" for i in range(n)],
    "tenure_months": tenure,
    "contract": contract,
    "internet_service": internet_service,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "tech_support": tech_support,
    "online_security": online_security,
    "paperless_billing": paperless_billing,
    "payment_method": payment_method,
    "senior_citizen": senior_citizen,
    "partner": partner,
    "num_support_calls": num_support_calls,
    "churn": churn_label,
})

# Inject a small amount of realistic messiness: a few missing values
missing_idx = rng.choice(n, size=25, replace=False)
df.loc[missing_idx, "total_charges"] = np.nan

df.to_csv("customer_churn.csv", index=False)
print(f"Saved customer_churn.csv with {len(df)} rows. Churn rate: {churn.mean():.1%}")
