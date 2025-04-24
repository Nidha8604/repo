import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Title
st.title("🎓 University Green Space & Wellbeing Predictor")

# Load dataset
df = pd.read_csv("university_student_wellbeing.csv")

# Features and target
features = ["NDVI Score", "Tree Density", "Green Space Area (sq.meters)",
            "Walking Distance (mins)", "Shade Coverage (%)", "Academic Stress Level"]
target = "Predicted Wellbeing Score"

# Sidebar inputs
st.sidebar.header("Input Parameters")

ndvi = st.sidebar.slider("NDVI Score", 0.3, 1.0, 0.6)
tree_density = st.sidebar.slider("Tree Density", 1, 10, 5)
area = st.sidebar.number_input("Green Space Area (sq. meters)", 100, 10000, 5000)
walk = st.sidebar.slider("Walking Distance (mins)", 1, 30, 10)
shade = st.sidebar.slider("Shade Coverage (%)", 0, 100, 50)
stress = st.sidebar.slider("Academic Stress Level", 1, 10, 5)

input_df = pd.DataFrame([[ndvi, tree_density, area, walk, shade, stress]],
                        columns=features)

# Train/test split
X = df[features]
y = df[target]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediction
prediction = model.predict(input_df)[0]
st.subheader("🌿 Predicted Wellbeing Score")
st.metric("Score", round(prediction, 2))

# Display input and preview
st.subheader("🔍 Your Input Summary")
st.dataframe(input_df)
