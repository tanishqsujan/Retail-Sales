import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("catboost_model.joblib")

st.set_page_config(page_title="Retail Sales Prediction", layout="wide")
st.title("📊 Retail Sales Prediction")
st.write("Predict weekly sales and analyze trends")

st.markdown("---")

st.header("Enter Inputs for Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    store = st.number_input("Store ID", min_value=1, max_value=50, value=1)
    holiday_flag = st.selectbox("Holiday Week", [0, 1])

with col2:
    temperature = st.number_input("Temperature", value=72.0)
    fuel_price = st.number_input("Fuel Price", value=3.50)

with col3:
    cpi = st.number_input("CPI", value=220.0)
    unemployment = st.number_input("Unemployment", value=7.5)

date = st.date_input("Select Date")

input_df = pd.DataFrame({
    "Store": [store],
    "Holiday_Flag": [holiday_flag],
    "Temperature": [temperature],
    "Fuel_Price": [fuel_price],
    "CPI": [cpi],
    "Unemployment": [unemployment],
    "Year": [date.year],
    "Month": [date.month],
    "Week": [date.isocalendar()[1]],
    "DayOfWeek": [date.weekday()],
    "DayOfYear": [date.timetuple().tm_yday],
})

st.subheader("Input Preview")
st.dataframe(input_df)

if st.button("Predict Weekly Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"**Predicted Weekly Sales: ${prediction:,.2f}**")

    st.subheader("Feature Importance")
    importances = model.get_feature_importance()
    feature_names = model.feature_names_ if hasattr(model, 'feature_names_') else input_df.columns

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.barplot(x=importances, y=feature_names, ax=ax1)
    ax1.set_title("Feature Importance")
    st.pyplot(fig1)

st.subheader("Predicted Sales Trend Over Time")

future_weeks = st.slider("Select number of future weeks to simulate", 4, 52, 12)

future_dates = pd.date_range(start=pd.Timestamp.today(), periods=future_weeks, freq='W')

future_df = pd.DataFrame({
    "Store": [store] * future_weeks,
    "Holiday_Flag": [holiday_flag] * future_weeks,
    "Temperature": [temperature] * future_weeks,
    "Fuel_Price": [fuel_price] * future_weeks,
    "CPI": [cpi] * future_weeks,
    "Unemployment": [unemployment] * future_weeks,
})

future_df["Year"] = future_dates.year
future_df["Month"] = future_dates.month
future_df["Week"] = future_dates.isocalendar().week
future_df["DayOfWeek"] = future_dates.weekday
future_df["DayOfYear"] = future_dates.dayofyear

future_predictions = model.predict(future_df)
future_df["Predicted_Sales"] = future_predictions
future_df["Date"] = future_dates

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(future_df["Date"], future_df["Predicted_Sales"], marker='o')
ax.set_title("Future Sales Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Weekly Sales")
plt.xticks(rotation=45)
st.pyplot(fig)

st.header("Upload Dataset to Visualize Sales Trends")
uploaded_file = st.file_uploader("Upload dataset", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "Date" in df.columns and "Weekly_Sales" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

        st.subheader("Weekly Sales Trend Over Time")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(df["Date"], df["Weekly_Sales"])
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Weekly Sales")
        ax2.set_title("Sales Trend")
        st.pyplot(fig2)

