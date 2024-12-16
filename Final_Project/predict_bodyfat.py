import pandas as pd
import joblib
import numpy as np

# Fungsi prediksi pipeline
def predict_pipeline(data_baru):
    # Muat model stacked
    stacked_model = joblib.load('saved_model/stacked_model.pkl')
    
    # Hitung fitur tambahan
    data_baru['BMI'] = data_baru['Weight'] / (data_baru['Height'] ** 2)
    data_baru['y_sr1'] = abs(abs(data_baru['Neck']) - ((data_baru['Abdomen'] - np.exp(data_baru['Height'])) / 1.1648)) - ((data_baru['Wrist'] * np.sin(data_baru['Sex'])) - (-1.7683))
    data_baru['y_sr2'] = ((data_baru['BMI'] - (np.sin(data_baru['Sex']) * (data_baru['Hip'] - (data_baru['Abdomen'] / 0.95054))) * np.cos(data_baru['Sex'] / data_baru['Height'])) - (np.sin(data_baru['Wrist']) + data_baru['Sex']))
    
    # Prediksi dengan model stacked
    final_prediction = stacked_model.predict(data_baru)
    return final_prediction

# Fungsi untuk mendapatkan input dari terminal
def get_user_input():
    print("Masukkan data berikut untuk prediksi Body Fat:")
    sex_input = input("Jenis Kelamin (Pria/Wanita): ").strip().lower()
    sex_value = 1 if sex_input == "pria" else 0  # Konversi 'Pria' ke 1 dan 'Wanita' ke 0
    data = {
        'Age': [int(input("Usia: "))],
        'Weight': [float(input("Berat Badan (kg): "))],
        'Height': [float(input("Tinggi Badan (m): "))],
        'Neck': [float(input("Lingkar Leher (cm): "))],
        'Chest': [float(input("Lingkar Dada (cm): "))],
        'Abdomen': [float(input("Lingkar Perut (cm): "))],
        'Hip': [float(input("Lingkar Pinggul (cm): "))],
        'Thigh': [float(input("Lingkar Paha (cm): "))],
        'Knee': [float(input("Lingkar Lutut (cm): "))],
        'Wrist': [float(input("Lingkar Pergelangan Tangan (cm): "))],
        'Sex': [sex_value]
    }
    return pd.DataFrame(data)

# Urutan fitur yang digunakan saat melatih model
feature_order = ['Sex', 'Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 
                 'Thigh', 'Knee', 'Wrist', 'BMI', 'y_sr1', 'y_sr2']

# Eksekusi program di terminal
if __name__ == "__main__":
    # Dapatkan input pengguna
    data_baru_df = get_user_input()
    
    # Hitung urutan fitur
    data_baru_df = data_baru_df.reindex(columns=feature_order, fill_value=0)
    
    # Prediksi
    hasil_prediksi = predict_pipeline(data_baru_df)
    
    # Tampilkan hasil prediksi
    print("\nHasil Prediksi Body Fat untuk Inputan Anda:", hasil_prediksi[0])
