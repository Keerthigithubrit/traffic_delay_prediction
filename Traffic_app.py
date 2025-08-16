# traffic_app.py
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from collections import Counter

# -------------------------
# 1. Load and Preprocess Data
# -------------------------

st.set_page_config(layout="wide")
st.title("🚦 Traffic Delay Prediction Dashboard")

try:
    data = pd.read_excel('traffic.xlsx', engine='openpyxl')
    st.success(f"✅ Loaded data with shape: {data.shape}")
    st.subheader("🔍 Preview of Cleaned Data")
    st.dataframe(data.head())
except FileNotFoundError:
    st.error("📂 File 'traffic.xlsx' not found. Please upload it to the project directory and reload the app.")
    st.stop()

# Convert object columns to datetime if possible
for col in data.columns:
    if data[col].dtype == 'object':
        try:
            data[col] = pd.to_datetime(data[col], format='%Y-%m-%d %H:%M:%S')
        except:
            pass

data = data.dropna()


# -------------------------
# 2. Define Features and Labels
# -------------------------

features = [
    'JamsDelay',
    'TrafficIndexLive',
    'JamsLengthInKms',
    'JamsCount',
    'TrafficIndexWeekAgo',
    'TravelTimeLivePer10KmsMins',
    'TravelTimeHistoricPer10KmsMins'
]

if any(col not in data.columns for col in features + ['MinsDelay']):
    st.error("🚫 Required columns are missing from the dataset.")
    st.stop()

X = data[features]

def classify_delay(x):
    if x <= 3:
        return 0  # Low
    elif x <= 7:
        return 1  # Medium
    else:
        return 2  # High

y = data['MinsDelay'].apply(classify_delay)

# -------------------------
# 3. Train Model
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

delay_labels = ['Low', 'Medium', 'High']
colors = ['green', 'orange', 'red']

# -------------------------
# 4. Real-Time Prediction
# -------------------------

st.subheader("🔮 Simulated Real-Time Prediction")

row = data.sample(1).iloc[0]
input_df = pd.DataFrame([row[features]])
pred = model.predict(input_df)[0]

st.markdown(f"""
- **City:** {row.get('City', 'Unknown')} ({row.get('Country', 'Unknown')})
- **Live Delay (mins):** {row['MinsDelay']}
- **Predicted Delay Level:** `{delay_labels[pred]}`
""")

# -------------------------
# 5. Folium Map
# -------------------------

st.subheader("🗺️ Map Visualization")

m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
folium.CircleMarker(
    location=[20.5937, 78.9629],
    radius=10,
    color=colors[pred],
    fill=True,
    fill_opacity=0.7,
    popup=f"Predicted: {delay_labels[pred]}"
).add_to(m)

st_folium(m, width=700, height=800)

# -------------------------
# 6. Evaluation Metrics
# -------------------------

with st.expander("📊 Show Model Evaluation"):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")

    if accuracy == 1.0:
        st.success("🎉 100% accuracy!")
    else:
        st.warning("Model is not 100% accurate.")

    st.write("**Test Set Class Distribution:**", {int(k): int(v) for k, v in Counter(y_test).items()})
    st.write("**Predicted Class Distribution:**", {int(k): int(v) for k, v in Counter(y_pred).items()})

    st.code(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=delay_labels, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4, 1))  # Slightly larger for clarity
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='coolwarm',
        xticklabels=delay_labels,
        yticklabels=delay_labels,
        cbar=False,
        square=True,
        linewidths=0.3,
        linecolor='gray',
        annot_kws={"size": 6},  # Font size for numbers inside boxes
        ax=ax
    )

    # Label axes
    ax.set_xlabel("Predicted", fontsize=6)
    ax.set_ylabel("Actual", fontsize=6)
    ax.set_title("Confusion Matrix", fontsize=7)

    # Set tick font sizes
    ax.tick_params(axis='both', labelsize=6)

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("🔎 Sample Predictions")
    for true_label, pred_label in zip(y_test[:10], y_pred[:10]):
        st.write(f"Actual: {delay_labels[true_label]}, Predicted: {delay_labels[pred_label]}")

# -------------------------
# 7. Feature Importance
# -------------------------

with st.expander("📌 Feature Importances"):
    importances = model.feature_importances_
    feature_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(feature_df.set_index("Feature"))

# -------------------------
# 8. User Custom Prediction
# -------------------------

with st.expander("🎛️ Try Your Own Inputs"):
    user_input = {
        feature: st.slider(
            label=feature,
            min_value=float(data[feature].min()),
            max_value=float(data[feature].max()),
            value=float(data[feature].mean())
        ) for feature in features
    }

    if st.button("Predict My Scenario"):
        input_df = pd.DataFrame([user_input])
        user_pred = model.predict(input_df)[0]
        st.success(f"🧠 Predicted Delay Level: **{delay_labels[user_pred]}**")

