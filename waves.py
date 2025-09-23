import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------
st.set_page_config(page_title="Ola Bike Ride Demand Forecasting", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Phase",
    ["Home", "Data Cleaning", "Clustering & Modelling", "Model Evaluation", "MAE Analysis", "Report", "Conclusion"]
)


# ---------------------------------------------------
# Home
# ---------------------------------------------------
if page == "Home":
    st.title("🚲 Wheels in Motion: Riding the Wave of Demand with Machine Learning")
    st.write("""
    This project demonstrates how **machine learning** can be applied to forecast Ola Bike ride demand in Bengaluru.

    **Phases of the project:**
    1. Data Cleaning  
    2. Clustering & Modelling  
    3. Model Evaluation  
    4. Report/Presentation
    5. Conclusion & Insights  
    """)

# ---------------------------------------------------
# Phase 1: Data Cleaning
# ---------------------------------------------------
elif page == "Data Cleaning":
    st.header("🧹 Phase 1: Data Cleaning")

    raw_df = pd.read_csv("Data/Bengaluru Ola.csv")
    cleaned_df = pd.read_csv("Data/Bengaluru_Ola_Bikes_Cleaned.csv")

    st.subheader("Raw Dataset Preview")
    st.dataframe(raw_df.head())

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(cleaned_df.head())

    # Missing values visualization
    st.subheader("Missing Values in Raw Dataset")
    missing_vals = raw_df.isnull().sum()
    st.bar_chart(missing_vals[missing_vals > 0])

    # Distribution before & after cleaning
    st.subheader("Ride Distance Distribution (Before vs After Cleaning)")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=raw_df, x="Ride Distance", bins=30, ax=ax[0], color="red")
    ax[0].set_title("Before Cleaning")
    sns.histplot(data=cleaned_df, x="Ride Distance", bins=30, ax=ax[1], color="green")
    ax[1].set_title("After Cleaning")
    st.pyplot(fig)

    st.subheader("Summary Statistics (Cleaned Data)")
    st.write(cleaned_df.describe())


# ---------------------------------------------------
# Phase 2: Clustering & Modelling
# ---------------------------------------------------
elif page == "Clustering & Modelling":
    st.header("📊 Phase 2: Clustering & Modelling")

    df = pd.read_csv("Data/Bengaluru_Ola_Bikes_Cleaned.csv")

    # Heatmap: Hour vs Day
    st.subheader("Heatmap of Demand (Hour vs Day)")
    if "Day" in df.columns and "Hour" in df.columns:
        pivot = df.pivot_table(index="Day", columns="Hour", values="Ride Distance", aggfunc="count")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

    # KMeans clustering
    st.subheader("KMeans Clustering: Hour vs Ride Distance")
    features = df[["Hour", "Ride Distance"]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    k = st.slider("Select number of clusters (k)", 2, 10, 4)
    model = KMeans(n_clusters=k, random_state=42)
    features["Cluster"] = model.fit_predict(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=features["Hour"], y=features["Ride Distance"],
        hue=features["Cluster"], palette="tab10", ax=ax
    )
    ax.set_title("Clusters of Demand")
    st.pyplot(fig)

    st.write("Clustered dataset preview:")
    st.dataframe(features.head())

    st.write("Cluster counts:")
    st.write(features["Cluster"].value_counts())

    # ---------------------------------------------------
    # Top Pickup Locations by Cluster (Bar Plot)
    # ---------------------------------------------------
    if "Pickup Location" in df.columns:
        st.subheader("Top Pickup Locations by Cluster")
        clustered_df = df.copy()
        clustered_df["Cluster"] = model.predict(X_scaled)

        # Count rides by pickup location & cluster
        top_locs = clustered_df.groupby(["Cluster", "Pickup Location"]).size().reset_index(name="Count")

        # Plot top 5 per cluster
        for c in sorted(top_locs["Cluster"].unique()):
            cluster_data = top_locs[top_locs["Cluster"] == c].nlargest(5, "Count")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x="Pickup Location", y="Count", data=cluster_data, ax=ax, palette="viridis")
            ax.set_title(f"Top Pickup Locations – Cluster {c}")
            ax.set_ylabel("Ride Count")
            ax.set_xlabel("Pickup Location")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)

        # ---------------------------------------------------
        # Heatmap of Pickup Location vs Cluster
        # ---------------------------------------------------
        st.subheader("Heatmap: Pickup Location vs Cluster")
        pivot_loc = top_locs.pivot_table(index="Pickup Location", columns="Cluster", values="Count", fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_loc, cmap="Blues", ax=ax)
        st.pyplot(fig)

# ---------------------------------------------------
# Phase 3: Model Evaluation
# ---------------------------------------------------
elif page == "Model Evaluation":
    st.header("🧪 Phase 3.1: Model Evaluation")

    # Example evaluation results (replace with your notebook results)
    spatial_data = {
        "Model": ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"],
        "MAE": [2.45, 1.98, 1.85, 1.80, 1.75],
        "RMSE": [3.60, 2.95, 2.80, 2.70, 2.65],
        "R²": [0.72, 0.81, 0.83, 0.85, 0.86]
    }
    temporal_data = {
        "Model": ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"],
        "MAE": [2.90, 2.30, 2.10, 2.00, 1.95],
        "RMSE": [4.10, 3.25, 3.05, 2.95, 2.90],
        "R²": [0.68, 0.78, 0.80, 0.82, 0.84]
    }

    cv_type = st.selectbox("Select Cross-Validation Type", ["Spatial", "Temporal"])
    eval_df = pd.DataFrame(spatial_data if cv_type == "Spatial" else temporal_data)

    st.subheader(f"Model Comparison ({cv_type} CV)")
    st.dataframe(eval_df)

    # Bar charts
    metrics = ["MAE", "RMSE", "R²"]
    for m in metrics:
        st.subheader(f"{m} Comparison")
        st.bar_chart(eval_df.set_index("Model")[m])

    # Best model highlight
    best_model = eval_df.sort_values("MAE").iloc[0]["Model"]
    st.success(f"✅ Best model for {cv_type} CV is **{best_model}**")


# ---------------------------------------------------
# Phase 3b: MAE Analysis
# ---------------------------------------------------
elif page == "MAE Analysis":
    st.header("📊 Phase 3.2: MAE Analysis")

    import json

    # Load evaluation results from JSON
    with open("Data/Phase3_evaluation_results.json", "r") as f:
        data = json.load(f)

    # Flatten JSON into a DataFrame
    rows = []
    for cv_strategy, models in data["results"].items():
        for model_name, metrics in models.items():
            folds = len(metrics["MAE"])
            for i in range(folds):
                rows.append({
                    "CV Strategy": cv_strategy,
                    "Model": model_name,
                    "Fold": i + 1,
                    "MAE": metrics["MAE"][i],
                    "RMSE": metrics["RMSE"][i],
                    "R2": metrics["R2"][i]
                })

    results = pd.DataFrame(rows)

    # Show raw flattened results
    st.subheader("Raw Evaluation Results")
    st.dataframe(results.head(20))

    # -----------------------------------------------
    # 1. Mean MAE by Model & CV Strategy
    # -----------------------------------------------
    st.subheader("Mean MAE by Model & CV Strategy")

    mean_mae = results.groupby(["Model", "CV Strategy"])["MAE"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=mean_mae, x="Model", y="MAE", hue="CV Strategy", palette="Set2", ax=ax)
    ax.set_title("Mean MAE by Model & CV Strategy")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

    # -----------------------------------------------
    # 2. MAE (Mean across folds)
    # -----------------------------------------------
    st.subheader("MAE (Mean across Folds)")

    mae_across_folds = results.groupby("Model")["MAE"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=mae_across_folds, x="Model", y="MAE", palette="viridis", ax=ax)
    ax.set_title("Mean MAE across Folds by Model")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

    # -----------------------------------------------
    # 3. MAE per Fold (lineplot for variability)
    # -----------------------------------------------
    st.subheader("MAE per Fold (Variability Check)")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=results, x="Fold", y="MAE",
        hue="Model", style="CV Strategy", marker="o", ax=ax
    )
    ax.set_title("MAE per Fold by Model & CV Strategy")
    st.pyplot(fig)


# ---------------------------------------------------
# Phase 4: Report (Presentation)
# ---------------------------------------------------
elif page == "Report":
    st.header("📑 Phase 4:Presentation")

    st.subheader("🎯 Objective")
    st.write("""
    The aim of Phase 4 was to present the results of the forecasting project,
    highlight key factors influencing bike ride demand, evaluate the chosen model’s performance,
    and discuss its limitations.
    """)

    st.subheader("📌 Factors Affecting Demand")
    st.markdown("""
    - **Time of Day** – Demand peaks during morning and evening commute hours.  
    - **Day of Week** – Weekdays and weekends show different demand patterns.  
    - **Location** – Business hubs generate more requests than residential areas.  
    - **Special Events & Holidays** – Cause sudden spikes/drops in demand.  
    - **External Conditions (not in dataset)** – Weather and traffic strongly influence demand.  
    """)

    st.subheader("🤖 Model Presentation")
    st.write("""
    The final forecasting model was selected from Phase 3,
    where boosting-based methods (Gradient Boosting, LightGBM, XGBoost) consistently outperformed others.
    The chosen model forecasts hourly demand by location with lower error compared to baselines.
    """)

    st.subheader("📊 Model Evaluation")
    st.markdown("""
    - **Metrics Used:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R².  
    - **Findings:** Boosting models achieved the lowest MAE (~0.34–0.36) and provided more accurate temporal forecasts than Random Forest & Linear Regression.  
    - **Insights:** Feature importance confirmed that hour of day and pickup location were the most significant predictors.  
    """)

    st.subheader("⚠️ Limitations")
    st.markdown("""
    - **Data Scope:** No real latitude/longitude or external variables like weather/traffic.  
    - **Feature Limitations:** Only time & location features, reducing model accuracy.  
    - **Forecast Horizon:** Only short-term hourly forecasts were tested.  
    """)


# ---------------------------------------------------
# Phase 4: Conclusion
# ---------------------------------------------------
elif page == "Conclusion":
    st.header("✅ Conclusion & Insights")
    st.success("""
 Machine learning can effectively forecast Ola Bike demand, with boosting models showing the best results.
    Including weather, events, and lag-based features would improve accuracy and robustness in real-world deployment.
    - Data cleaning removed noise (missing values, outliers).  
    - KMeans clustering revealed demand patterns by hour and ride distance.  
    - Across models, **LightGBM** achieved the best performance (lowest MAE/RMSE, highest R²).  
    - Spatial CV showed models generalize well to unseen pickup locations.  
    - Temporal CV demonstrated predictive strength for future time periods.  
    """)