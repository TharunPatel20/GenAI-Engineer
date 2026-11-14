"""Project 01: Python Foundations - analyze a small loan dataset.

Run:  python analyze_loans.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv("loans_sample.csv")
print("First 5 rows:")
print(data.head())

# Step 2: Basic counts
print("\nLoan status counts (Y = approved, N = rejected):")
print(data["Loan_Status"].value_counts())

# Step 3: Summary statistics for numeric columns
print("\nSummary of numeric columns:")
print(data.describe())

# Step 4: Create a combined income column, then drop the originals
data["Total_Income"] = data["Applicant_Income"] + data["Coapplicant_Income"]
data = data.drop(columns=["Applicant_Income", "Coapplicant_Income"])

# Step 5: Encode a text column as numbers (a simple mapping)
data["Married_Code"] = data["Married"].map({"No": 0, "Yes": 1})

# Step 6: Average income by approval status
print("\nAverage total income by loan status:")
print(data.groupby("Loan_Status")["Total_Income"].mean().round(0))

# Step 7: Bar chart of approved vs rejected
counts = data["Loan_Status"].value_counts().reindex(["Y", "N"])
plt.bar(["Approved (Y)", "Rejected (N)"], counts.values, color=["#2a9d8f", "#e76f51"])
plt.ylabel("Number of applications")
plt.title("Loan applications by status")
plt.tight_layout()
plt.savefig("loan_status_chart.png")
print("\nSaved chart to loan_status_chart.png")
