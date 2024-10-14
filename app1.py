import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Judul aplikasi
st.title("Aplikasi Prediksi Harga Rumah")

# Load dataset
@st.cache_data
def load_data():
    file_path = "D:/MachineLearning/house_price_regression_dataset.csv"  # Pastikan path ini sesuai
    data = pd.read_csv(file_path)
    return data

data = load_data()

# Pastikan kolom ada
if 'House_Price' not in data.columns:
    st.error("Kolom 'House_Price' tidak ditemukan dalam dataset.")
    st.stop()

# Definisikan variabel independen dan dependen
X = data.drop(columns='House_Price')
y = data['House_Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
model = LinearRegression()
model.fit(X_train, y_train)

# Input dari pengguna
st.sidebar.header("Masukkan Data Properti")
square_footage = st.sidebar.number_input("Luas Bangunan (Square Footage)", value=float(X['Square_Footage'].mean()))
num_bedrooms = st.sidebar.slider("Jumlah Kamar Tidur", int(X['Num_Bedrooms'].min()), int(X['Num_Bedrooms'].max()), int(X['Num_Bedrooms'].mean()))
num_bathrooms = st.sidebar.slider("Jumlah Kamar Mandi", int(X['Num_Bathrooms'].min()), int(X['Num_Bathrooms'].max()), int(X['Num_Bathrooms'].mean()))
year_built = st.sidebar.slider("Tahun Dibangun", int(X['Year_Built'].min()), int(X['Year_Built'].max()), int(X['Year_Built'].mean()))
lot_size = st.sidebar.number_input("Ukuran Lahan (Lot Size)", value=float(X['Lot_Size'].mean()))
garage_size = st.sidebar.slider("Ukuran Garasi", int(X['Garage_Size'].min()), int(X['Garage_Size'].max()), int(X['Garage_Size'].mean()))
neighborhood_quality = st.sidebar.slider("Kualitas Lingkungan", int(X['Neighborhood_Quality'].min()), int(X['Neighborhood_Quality'].max()), int(X['Neighborhood_Quality'].mean()))

# Buat dataframe baru
new_data = pd.DataFrame({
    'Square_Footage': [square_footage],
    'Num_Bedrooms': [num_bedrooms],
    'Num_Bathrooms': [num_bathrooms],
    'Year_Built': [year_built],
    'Lot_Size': [lot_size],
    'Garage_Size': [garage_size],
    'Neighborhood_Quality': [neighborhood_quality]
})

# Prediksi
prediction = model.predict(new_data)
predicted_price = prediction[0]

# Tampilkan hasil
st.subheader("Hasil Prediksi Harga Rumah")
st.write(f"Harga Prediksi: Rp {predicted_price:,.2f}")

# Data pengguna
st.subheader("Data Properti yang Anda Masukkan")
st.write(new_data)
