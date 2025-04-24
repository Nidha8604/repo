import streamlit as st
import pandas as pd
from sklearn.ensemble 
import RandomForestRegressor

st.set_page_config(page_title="Green Wellbeing Predictor", layout="centered")

st.title("🌿 Green Campus Wellbeing Predictor")

# Upload dataset
uploaded_file = st.file_uploader("university_student_wellbeing.csv", type="csv")

if uploaded_file is not None:
    # Read the dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Data loaded successfully!")
    st.write("Sample of the dataset:")
    st.dataframe(df.head())

    # Features and target
    features = ["NDVI Score", "Tree Density", "Green Space Area (sq.meters)",
                "Walking Distance (mins)", "Shade Coverage (%)", "Sleep Hours"]
    target = "Predicted Wellbeing Score"

    # Train model
    X = df[features]
    y = df[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Inputs
    st.subheader("🎯 Input Green Space Features")

    ndvi = st.slider("NDVI Score", 0.0, 1.0, 0.75)
    tree_density = st.slider("Tree Density", 0, 15, 7)
    area = st.number_input("Green Space Area (sq. meters)", 100, 10000, 5000)
    walk = st.slider("Walking Distance (mins)", 0, 30, 10)
    shade = st.slider("Shade Coverage (%)", 0.0, 100.0, 60.0)
    sleep = st.slider("Sleep Hours (per day)", 0, 12, 7)

    input_df = pd.DataFrame([[ndvi, tree_density, area, walk, shade, sleep]],
                            columns=features)

    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.subheader("🧠 Predicted Wellbeing Score")
    st.metric("Wellbeing Score", f"{prediction:.2f}")

else:
    st.info("📤 Please upload a dataset to begin.")
