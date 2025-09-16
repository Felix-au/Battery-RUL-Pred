import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------
# Load all trained models
# --------------------------
models = {
    "Linear Regression": joblib.load("linear_regression_model.pkl"),
    "KNN": joblib.load("knn_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl"),
    "XGBoost": joblib.load("xgboost_model.pkl"),
    "SVR (Default)": joblib.load("svr_default_model.pkl"),
    "SVR (Tuned)": joblib.load("svr_tuned_model.pkl"),
    "Gradient Boosting": joblib.load("gradient_boosting_model.pkl"),
}

st.title("🔋 Battery RUL Prediction (Multiple Models)")

# --------------------------
# File uploader
# --------------------------
uploaded_file = st.file_uploader("Upload your Battery dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("📂 Uploaded Data Preview")
    st.dataframe(df.head())

    # Drop missing values
    df = df.dropna()

    # Split features & target
    if "RUL" not in df.columns:
        st.error("❌ The dataset must contain an 'RUL' column as the target.")
    else:
        X = df.drop("RUL", axis=1)
        y = df["RUL"]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --------------------------
        # Evaluate all models
        # --------------------------
        results = []
        st.subheader("📊 Model Performance")

        for name, model in models.items():
            y_pred = model.predict(X_scaled)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            results.append((name, rmse, r2))

            # Plot actual vs predicted
            st.write(f"### {name}")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(y.values[:100], label="Actual RUL", marker="o")
            ax.plot(y_pred[:100], label=f"Predicted RUL - {name}", marker="x")
            ax.set_title(f"{name}: Actual vs Predicted RUL")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Remaining Useful Life")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # --------------------------
        # Show comparison table
        # --------------------------
        results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R² Score"])
        st.subheader("📌 Final Comparison")
        st.dataframe(results_df.sort_values(by="RMSE"))
