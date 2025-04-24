import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Excel file
@st.cache_data
def load_data():
    df = pd.read_excel("Bookc2excek.xlsx", sheet_name="Sheet1")
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df = pd.get_dummies(df, columns=["Gender", "Green Space Type"], drop_first=True)
    return df

df = load_data()

# Feature selection
features = ["NDVI Score", "Tree Density", "Green Space Area (sq.meters)",
            "Walking Distance (mins)", "Shade Coverage (%)"]
X = df[features]
y = df["Predicted Wellbeing Score"]

# Train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_scaled, y)

# UI
st.title("🌿 Green Space & Student Wellbeing Predictor")

ndvi = st.slider("NDVI Score", 0.0, 1.0, 0.7)
tree_density = st.slider("Tree Density", 0, 10, 5)
area = st.number_input("Green Space Area (sq. meters)", 1000, 10000, 5000)
walk = st.slider("Walking Distance (mins)", 0, 30, 10)
shade = st.slider("Shade Coverage (%)", 0, 100, 50)

# Prediction
input_df = pd.DataFrame([[ndvi, tree_density, area, walk, shade]],
                        columns=features)

scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)

st.success(f"🧠 Predicted Wellbeing Score: {round(prediction[0], 2)}")
