import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
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
        # Model Selection
        # --------------------------
        selected_models = st.multiselect(
            "Select Models to Evaluate",
            list(models.keys()),
            default=[]  # none selected by default
        )

        if selected_models:
            results = []
            st.subheader("📊 Model Performance")

            for name in selected_models:
                model = models[name]
                y_pred = model.predict(X_scaled)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                r2 = r2_score(y, y_pred)
                results.append((name, rmse, r2))

                # Interactive Plot (first 100 samples)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=y.values[:100], mode="lines+markers", name="Actual RUL"
                ))
                fig.add_trace(go.Scatter(
                    y=y_pred[:100], mode="lines+markers", name=f"Predicted RUL - {name}"
                ))
                fig.update_layout(
                    title=f"{name}: Actual vs Predicted RUL",
                    xaxis_title="Sample Index",
                    yaxis_title="Remaining Useful Life",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

            # --------------------------
            # Show comparison table
            # --------------------------
            results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R² Score"])
            results_df = results_df.sort_values(by=["RMSE", "R² Score"], ascending=[True, False])

            st.subheader("📌 Final Comparison")
            st.dataframe(results_df)

            # Highlight best model
            best_model = results_df.iloc[0]
            st.success(f"🏆 Best Model: **{best_model['Model']}** "
                       f"(RMSE: {best_model['RMSE']:.2f}, R²: {best_model['R² Score']:.2f})")

            # --------------------------
            # Download Results
            # --------------------------
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="model_comparison_results.csv",
                mime="text/csv",
            )
        else:
            st.info("ℹ️ Please select at least one model to evaluate.")
