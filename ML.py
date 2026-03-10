import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="AI Cyclone Prediction", layout="wide")

# ---------------------- BLACK BACKGROUND ----------------------
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}
h1, h2, h3, h4 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

st.title("🌪 AI-Based Tropical Cyclone Prediction System")

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/hafee/OneDrive/Desktop/Desktop/AI-Based Tropical Cyclone Forecasting/cyclone_dataset.csv")
    df.drop(['Pre_existing_Disturbance','Vorticity'], axis=1, inplace=True)
    return df

df = load_data()

# Features & Target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ---------------------- MODEL TRAINING ----------------------
@st.cache_resource
def train_model(X, y):

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='accuracy'
    )

    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(x_test)

    return best_model, x_test, y_test, y_pred

with st.spinner("⚙ Model Training Running... Please wait"):
    best_rf, x_test, y_test, y_pred = train_model(X, y)

# ---------------------- TABS ----------------------
tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Reports & Visualization"])

# ============================================================
# TAB 1 - PREDICTION
# ============================================================
with tab1:
    st.header("Enter Atmospheric Conditions")

    feature_inputs = []
    cols = st.columns(2)

    for i, col_name in enumerate(X.columns):

        min_val = float(X[col_name].min())
        max_val = float(X[col_name].max())
        mean_val = float(X[col_name].mean())

        with cols[i % 2]:
            value = st.slider(
                col_name,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )
            feature_inputs.append(value)

    if st.button("Predict Cyclone"):

        # ✅ FIXED HERE
        input_data = np.array([feature_inputs])

        prediction = best_rf.predict(input_data)[0]
        probability = best_rf.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error("🌪 Cyclone Likely to Form")
        else:
            st.success("✅ No Cyclone Formation")

        st.info(f"📊 Probability of Cyclone Formation: {probability*100:.2f}%")
        st.progress(float(probability))

# ============================================================
# TAB 2 - REPORTS
# ============================================================
with tab2:
    st.header("Model Reports & Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Model Accuracy")
    st.write("Accuracy Score:", accuracy_score(y_test, y_pred))

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

    # ----------------- VISUALIZATIONS -----------------

    st.subheader("Sea Surface Temperature Distribution")
    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.hist(df['Sea_Surface_Temperature'], bins=10)
    ax1.set_title("Sea Surface Temperature Distribution")
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Wind Shear vs Cyclone")
    fig3, ax3 = plt.subplots(figsize=(6,4))
    sns.boxplot(x='Cyclone', y='Wind_Shear', data=df, ax=ax3)
    st.pyplot(fig3)

    st.subheader("Cyclone Distribution")
    fig4, ax4 = plt.subplots(figsize=(6,4))
    sns.countplot(x='Cyclone', data=df, ax=ax4)
    st.pyplot(fig4)