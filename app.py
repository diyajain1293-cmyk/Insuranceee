
import streamlit as st
import pandas as pd
import numpy as np
import io

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Insurance Policy Status – Dashboard & ML", layout="wide")

# ------------------------ Helpers ------------------------
@st.cache_data
def load_data(default_path="Insurance.csv"):
    try:
        df = pd.read_csv(default_path)
    except Exception:
        df = pd.DataFrame()
    return df

def basic_clean(df):
    df = df.copy()
    # Remove thousands separators (commas) that appear in numeric-looking columns
    df = df.replace(',', '', regex=True)
    # Coerce obvious numeric columns if possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            # keep as object if cannot be numeric
            pass
    # Standardize column names
    df.columns = df.columns.str.strip()
    return df

def preprocess_for_model(df, target_col="POLICY_STATUS"):
    df = df.copy()
    df = basic_clean(df)

    # Keep a copy of original for charts
    original = df.copy()

    # Drop high-cardinality identifiers if present
    for drop_col in ["POLICY_NO", "PI_NAME"]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if df[col].mode().size else "Unknown")

    # Encode categoricals (Label Encoding keeps columns simple)
    encoders = {}
    for col in df.columns:
        if df[col].dtype == "object" and col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Encode target if needed
    y = df[target_col].astype(str) if df[target_col].dtype == "object" else df[target_col]
    if y.dtype == "object":
        y_enc = LabelEncoder()
        y = y_enc.fit_transform(y)
        encoders[target_col] = y_enc
    X = df.drop(columns=[target_col])

    return X, y, encoders, original

def train_models(X, y, cv_splits=5, random_state=42):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Gradient Boosted": GradientBoostingClassifier(random_state=random_state)
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    results = {}
    for name, model in models.items():
        # Cross-validated accuracy as "Training Accuracy" proxy
        cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
        model.fit(X, y)
        y_pred = model.predict(X)
        # Some classifiers may not have predict_proba; all here do
        y_prob = model.predict_proba(X)[:, 1]
        results[name] = {
            "Training Accuracy": cv_acc,
            "Testing Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1 Score": f1_score(y, y_pred),
            "ROC AUC": roc_auc_score(y, y_prob),
            "Confusion Matrix": confusion_matrix(y, y_pred),
            "Model": model,
            "y_prob": y_prob
        }
    return results

def plot_plain_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")
    # Create a white canvas and place numbers
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(0.25 + j*0.5, 0.75 - i*0.5, str(cm[i, j]), ha="center", va="center", fontsize=16)
    ax.set_title(title)
    return fig

def plot_roc(y, y_prob, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, linewidth=2, color="black")
    ax.plot([0,1], [0,1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    return fig

def top_feature_importances(model, feature_names, k=10):
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        return importances.head(k)
    return pd.Series(dtype=float)

# ------------------------ Data ------------------------
st.title("📊 Insurance Policy Status – Streamlit Cloud Dashboard")

st.write("Upload nothing to start with the default `Insurance.csv` (if present). "
         "Or upload your own dataset in the **Predict** tab.")

df_loaded = load_data()
if df_loaded.empty:
    st.warning("No default dataset found in this app package. Please ensure `Insurance.csv` is in the app root or upload data in the Predict tab.")
else:
    st.success("Default dataset loaded from `Insurance.csv`.")

# ------------------------ Tabs ------------------------
tab1, tab2, tab3 = st.tabs(["📈 Insights Dashboard", "🤖 Train & Evaluate Models", "📤 Predict on New Data"])

# ------------------------ Tab 1: Insights ------------------------
with tab1:
    st.subheader("Interactive Insights")
    if df_loaded.empty:
        st.info("Please load data first.")
    else:
        df = basic_clean(df_loaded)

        # Build filters
        status_col = "POLICY_STATUS" if "POLICY_STATUS" in df.columns else None
        reason_col = "REASON_FOR_CLAIM" if "REASON_FOR_CLAIM" in df.columns else None

        cols = st.columns(2)
        with cols[0]:
            status_vals = sorted(df[status_col].dropna().unique()) if status_col else []
            selected_status = st.multiselect("Filter by Policy Status", status_vals, default=status_vals)
        with cols[1]:
            reason_vals = sorted(df[reason_col].fillna("Missing").unique()) if reason_col else []
            selected_reason = st.multiselect("Filter by Reason for Claim", reason_vals, default=reason_vals)

        # Apply filters
        dff = df.copy()
        if status_col and selected_status:
            dff = dff[dff[status_col].isin(selected_status)]
        if reason_col and selected_reason:
            # Replace NaN with 'Missing' for filtering
            dff = dff.copy()
            dff[reason_col] = dff[reason_col].fillna("Missing")
            dff = dff[dff[reason_col].isin(selected_reason)]

        # Ensure numeric conversion for charts
        for col in ["SUM_ASSURED", "PI_AGE", "PI_ANNUAL_INCOME"]:
            if col in dff.columns:
                try:
                    dff[col] = pd.to_numeric(dff[col])
                except Exception:
                    pass

        # 5 different charts with actionable insights
        c1, c2 = st.columns(2)

        # Chart 1: Approval rate by state (top 10 states)
        if status_col and "PI_STATE" in dff.columns:
            state_rate = (dff.groupby("PI_STATE")[status_col]
                          .apply(lambda s: (s == s.mode().iloc[0]).mean() if s.size else 0))
            top_states = state_rate.sort_values(ascending=False).head(10)
            fig, ax = plt.subplots()
            top_states.plot(kind="bar", ax=ax)
            ax.set_title("Approval-Mode Rate by State (Top 10)")
            ax.set_ylabel("Rate")
            ax.set_xlabel("PI_STATE")
            c1.pyplot(fig, use_container_width=True)

        # Chart 2: Median sum assured by POLICY_STATUS
        if status_col and "SUM_ASSURED" in dff.columns:
            med_sa = dff.groupby(status_col)["SUM_ASSURED"].median().sort_values(ascending=False)
            fig, ax = plt.subplots()
            med_sa.plot(kind="bar", ax=ax)
            ax.set_title("Median Sum Assured by Policy Status")
            ax.set_ylabel("Median Sum Assured")
            c2.pyplot(fig, use_container_width=True)

        # Chart 3: Age vs approval rate curve (binned)
        if status_col and "PI_AGE" in dff.columns:
            bins = pd.cut(dff["PI_AGE"], bins=[0,30,40,50,60,70,80,100], include_lowest=True)
            rate_by_age = dff.groupby(bins)[status_col].apply(lambda s: (s == s.mode().iloc[0]).mean())
            fig, ax = plt.subplots()
            rate_by_age.plot(marker="o", ax=ax)
            ax.set_title("Approval-Mode Rate by Age Band")
            ax.set_ylabel("Rate")
            ax.set_xlabel("Age Band")
            st.pyplot(fig, use_container_width=True)

        # Chart 4: Payment mode distribution with approval skew
        if status_col and "PAYMENT_MODE" in dff.columns:
            pm = (dff.groupby(["PAYMENT_MODE", status_col]).size()
                    .reset_index(name="count"))
            pm_pivot = pm.pivot(index="PAYMENT_MODE", columns=status_col, values="count").fillna(0)
            pm_pivot = pm_pivot.loc[pm_pivot.sum(axis=1).sort_values(ascending=False).index]
            fig, ax = plt.subplots()
            pm_pivot.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title("Payment Mode Distribution by Policy Status")
            ax.set_xlabel("Payment Mode")
            ax.set_ylabel("Count")
            st.pyplot(fig, use_container_width=True)

        # Chart 5: Occupation – Top 12 categories with approval counts
        if status_col and "PI_OCCUPATION" in dff.columns:
            occ_counts = dff.groupby(["PI_OCCUPATION", status_col]).size().reset_index(name="count")
            top_occ = occ_counts.groupby("PI_OCCUPATION")["count"].sum().sort_values(ascending=False).head(12).index
            occ_counts = occ_counts[occ_counts["PI_OCCUPATION"].isin(top_occ)]
            pivot = occ_counts.pivot(index="PI_OCCUPATION", columns=status_col, values="count").fillna(0)
            fig, ax = plt.subplots()
            pivot.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title("Top Occupations by Policy Status")
            ax.set_xlabel("Occupation")
            ax.set_ylabel("Count")
            st.pyplot(fig, use_container_width=True)

        st.caption("All charts respect the multiselect filters for Policy Status and Reason for Claim.")

# ------------------------ Tab 2: Models ------------------------
with tab2:
    st.subheader("Apply 3 Algorithms & Generate Metrics")
    if df_loaded.empty:
        st.info("Please load data first.")
    else:
        X, y, encoders, original = preprocess_for_model(df_loaded, target_col="POLICY_STATUS")
        if st.button("Run Models (5-fold CV)"):
            results = train_models(X, y, cv_splits=5)

            # Metrics table
            metrics = pd.DataFrame({name: {
                "Training Accuracy": r["Training Accuracy"],
                "Testing Accuracy": r["Testing Accuracy"],
                "Precision": r["Precision"],
                "Recall": r["Recall"],
                "F1 Score": r["F1 Score"],
                "ROC AUC": r["ROC AUC"]
            } for name, r in results.items()}).T
            st.dataframe(metrics.style.format(precision=4), use_container_width=True)

            # Confusion matrices (plain white – no color)
            st.markdown("**Confusion Matrices (no color):**")
            cols = st.columns(3)
            i = 0
            for name, r in results.items():
                fig = plot_plain_confusion_matrix(r["Confusion Matrix"], f"{name}")
                cols[i%3].pyplot(fig, use_container_width=True)
                i += 1

            # Individual ROC curves + Combined
            st.markdown("**ROC Curves:**")
            cols = st.columns(3)
            for idx, (name, r) in enumerate(results.items()):
                fig = plot_roc(y, r["y_prob"], f"ROC – {name}")
                cols[idx%3].pyplot(fig, use_container_width=True)

            # Combined ROC
            fig, ax = plt.subplots()
            for name, r in results.items():
                fpr, tpr, _ = roc_curve(y, r["y_prob"])
                ax.plot(fpr, tpr, label=f"{name} (AUC={r['ROC AUC']:.2f})")
            ax.plot([0,1],[0,1],'k--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Combined ROC Curves")
            ax.legend()
            st.pyplot(fig, use_container_width=True)

            # Feature importances from best (by ROC AUC)
            best_name = max(results, key=lambda k: results[k]["ROC AUC"])
            best_model = results[best_name]["Model"]
            fi = top_feature_importances(best_model, X.columns, k=10)
            if not fi.empty:
                st.markdown(f"**Top 10 Feature Importances – {best_name}:**")
                st.bar_chart(fi.sort_values())

# ------------------------ Tab 3: Predict ------------------------
with tab3:
    st.subheader("Upload New Dataset & Predict Policy Status")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        new_df = pd.read_csv(uploaded)
        st.write("Preview:", new_df.head())

        # Fit models on current default dataset for prediction
        if df_loaded.empty:
            st.warning("Cannot train model because base dataset is missing.")
        else:
            X, y, encoders, original = preprocess_for_model(df_loaded, target_col="POLICY_STATUS")
            results = train_models(X, y, cv_splits=5)
            # Use best ROC AUC model
            best_name = max(results, key=lambda k: results[k]["ROC AUC"])
            model = results[best_name]["Model"]

            # Prepare new data with same transforms
            new_proc = new_df.copy()
            new_proc = basic_clean(new_proc)
            # Handle missing
            for col in new_proc.columns:
                if new_proc[col].dtype in ["float64", "int64"]:
                    new_proc[col] = new_proc[col].fillna(new_proc[col].median())
                else:
                    new_proc[col] = new_proc[col].fillna(new_proc[col].mode()[0] if new_proc[col].mode().size else "Unknown")
            # Drop id-like cols
            for drop_col in ["POLICY_NO", "PI_NAME"]:
                if drop_col in new_proc.columns:
                    new_proc = new_proc.drop(columns=[drop_col])

            # Align columns to training X
            aligned = {}
            for col in X.columns:
                if col in new_proc.columns:
                    # Numerical or encoded categorical (attempt numeric conversion)
                    try:
                        aligned[col] = pd.to_numeric(new_proc[col])
                    except Exception:
                        aligned[col] = new_proc[col]
                else:
                    aligned[col] = np.nan  # will fill median next
            aligned_df = pd.DataFrame(aligned)
            for col in aligned_df.columns:
                if aligned_df[col].dtype in ["float64", "int64"]:
                    aligned_df[col] = aligned_df[col].fillna(aligned_df[col].median())
                else:
                    aligned_df[col] = aligned_df[col].fillna(method="ffill").fillna(method="bfill")

            preds = model.predict(aligned_df)
            # If we have original encoder for target, inverse transform for labels
            label = encoders.get("POLICY_STATUS", None)
            if label is not None:
                preds_readable = label.inverse_transform(preds)
            else:
                preds_readable = preds

            out = new_df.copy()
            out["PREDICTED_POLICY_STATUS"] = preds_readable
            st.write("Predictions:", out.head())

            # Download
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions CSV", data=csv, file_name="predictions_with_policy_status.csv", mime="text/csv")

st.caption("Tip: Keep requirements minimal (no pinned versions) for Streamlit Cloud compatibility.")
