import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 50000

print("Generating synthetic data...")

# 1. Metadata
user_ids = [f"user_{100000 + i}" for i in range(n_samples)]
device_age_days = np.random.randint(30, 1000, size=n_samples)  # Device age in days
# High sim swap count is a known instability indicator
sim_swap_count_3m = np.random.choice(
    [0, 1, 2, 3], size=n_samples, p=[0.85, 0.10, 0.04, 0.01]
)

# 2. Mobile Money Feautures
# Average monthly inbound P2P or merchant payments (KES or local currency equivalent)
avg_monthly_inflow = np.random.exponential(scale=15000, size=n_samples) + 2000
# Coefficient of variation of inflow (higher means less predictable income)
inflow_volatility = np.random.uniform(0.05, 0.80, size=n_samples)
# Utility bill payment ratio (percentage of bills paid on time over mobile money
utility_bill_payment_ratio = np.random.beta(a=5, b=2, size=n_samples)

# 3. Telco Telemetry Features
# Airtime top-up frequency (number of top-ups per month)
airtime_topup_freq_pm = np.random.negative_binomial(
    n=5, p=0.2, size=n_samples
) + 1
# Averrage airtime purchase amount (KES or local currency equivalent)
avg_airtime_topup_val = np.random.lognormal(mean=4.5, sigma=0.5, size=n_samples)

# 4. Synthesizing the Ground Truth (Probability of default)
latent_risk = (
    (sim_swap_count_3m * 1.2)
    + (inflow_volatility * 2.5)
    - (utility_bill_payment_ratio * 2.0)
    - (np.log1p(avg_monthly_inflow) * 0.8)
    + (airtime_topup_freq_pm * 0.02)
    + np.random.normal(
        0, 1.5, size=n_samples
    )
)

# Convert latent risk into a probability between 0 and 1 using a sigmoid function
probability_of_default = 1 / (1 + np.exp(-latent_risk))
# Define default event (1 = Defaulted within 30 days, 0 = Paid back)
# Adjust threshold to target roughly a ~6-8% baseline default rate before model filtering
default_within_30_days = np.where(probability_of_default > 0.72, 1, 0)

# 5. Combine into a DataFrame
df = pd.DataFrame(
    {
        "user_id": user_ids,
        "device_age_days": device_age_days,
        "sim_swap_count_3m": sim_swap_count_3m,
        "avg_monthly_inflow": np.round(avg_monthly_inflow, 2),
        "inflow_volatility": np.round(inflow_volatility, 3),
        "utility_bill_payment_ratio": np.round(utility_bill_payment_ratio, 3),
        "airtime_topup_freq_pm": airtime_topup_freq_pm,
        "avg_airtime_topup_val": np.round(avg_airtime_topup_val, 2),
        "default_within_30_days": default_within_30_days
    }
)

output_dir = Path('./data')

# Save dataset to a CSV file for the modeling phase
df.to_csv(output_dir / "thin_file_credit_data.csv", index=False)

# Quick Sanity Check Printout
print("\nDataset Generation Complete!")
print(f"Dataset Shape: {df.shape}")
print(
    f"Baseline Default Rate (NPL): {df['default_within_30_days'].mean() * 100:.2f}%"
)
print("\nFirst 3 rows of the data:")
print(df.head(3).to_string(index=False))