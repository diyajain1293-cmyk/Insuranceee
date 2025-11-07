# Insurance Policy Status – Streamlit Cloud App

This app provides:
- Interactive insights with 5 actionable charts (all charts filter by **Policy Status** and **Reason for Claim**).
- A **Models** tab to run Decision Tree, Random Forest, and Gradient Boosting with 5-fold CV — showing metrics, plain (no-color) confusion matrices, and ROC curves (individual + combined).
- A **Predict** tab to upload a new CSV and download predictions (`PREDICTED_POLICY_STATUS`).

## How to deploy on Streamlit Cloud
1. Push these files (no folders) to your GitHub repo:
   - `app.py`
   - `requirements.txt`
   - `Insurance.csv` (optional but recommended as a default dataset)
2. Create a new Streamlit app from your repo and set **Main file path** to `app.py`.

## Notes
- Requirements do not pin versions to reduce dependency conflicts.
- If you don't include `Insurance.csv`, simply use the **Predict** tab to upload data.
