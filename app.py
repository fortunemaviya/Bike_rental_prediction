import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# -------------------------------
# Function to create dataset
# -------------------------------
def create_multivariate_dataset(dataset, window_size=7, target_col=0):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i:i+window_size, :])
        y.append(dataset[i + window_size, target_col])
    return np.array(X), np.array(y)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Bike Rentals Forecasting", page_icon="🚴", layout="wide")
st.title("🚴 Bike Rental Prediction with LSTM (Multivariate)")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your bike rental CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure required columns exist
    features = ['cnt','hum','temp','windspeed','weathersit']
    if not all(col in df.columns for col in features):
        st.error(f"Uploaded file must contain columns: {features}")
        st.stop()

    st.subheader("📊 Data Preview")
    st.dataframe(df[features].head())

    # Scale data
    data = df[features].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Create sliding window dataset
    window_size = 7
    X, y = create_multivariate_dataset(data_scaled, window_size, target_col=0)
    st.write(f"✅ Dataset created with shape: X={X.shape}, y={y.shape}")

    # Train/test split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(64, input_shape=(window_size, X.shape[2]), return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    # Train model
    with st.spinner("Training LSTM model (30 epochs)..."):
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=16,
            validation_data=(X_test, y_test),
            verbose=0
        )
    st.success("✅ Model training complete!")

    # Predict
    y_pred = model.predict(X_test)

    # Invert scaling for visualization
    y_test_full, y_pred_full = [], []
    for i in range(len(y_test)):
        last_features = X_test[i][-1][1:]   # all features except rentals
        y_test_row = np.concatenate(([y_test[i]], last_features))
        y_pred_row = np.concatenate(([y_pred[i][0]], last_features))
        y_test_full.append(y_test_row)
        y_pred_full.append(y_pred_row)

    y_test_full = np.array(y_test_full)
    y_pred_full = np.array(y_pred_full)

    y_test_inv = scaler.inverse_transform(y_test_full)[:,0]
    y_pred_inv = scaler.inverse_transform(y_pred_full)[:,0]

    # Build results dataframe
    results_df = df.iloc[-len(y_test):][features].copy()
    results_df["Predicted_cnt"] = y_pred_inv

    st.subheader("📈 Actual vs Predicted Bike Rentals")
    st.dataframe(results_df.head(20))

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_test_inv, label="Actual cnt")
    ax.plot(y_pred_inv, label="Predicted cnt")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Please upload a dataset to start.")
