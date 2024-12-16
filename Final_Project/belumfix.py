from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Inisialisasi Flask app
app = Flask(__name__)

# Memuat model stacked yang sudah dilatih
stacked_model = joblib.load('stacked_model.pkl')

# Fungsi pipeline prediksi
def predict_pipeline(data_baru):
    # Menambahkan fitur tambahan yang diperlukan
    data_baru['BMI'] = data_baru['Weight'] / (data_baru['Height'] ** 2)
    data_baru['y_sr1'] = abs(abs(data_baru['Neck']) - ((data_baru['Abdomen'] - np.exp(data_baru['Height'])) / 1.1648)) - ((data_baru['Wrist'] * np.sin(data_baru['Sex'])) - (-1.7683))
    data_baru['y_sr2'] = ((data_baru['BMI'] - (np.sin(data_baru['Sex']) * (data_baru['Hip'] - (data_baru['Abdomen'] / 0.95054))) * np.cos(data_baru['Sex'] / data_baru['Height'])) - (np.sin(data_baru['Wrist']) + data_baru['Sex']))

    # Prediksi menggunakan model stacked
    final_prediction = stacked_model.predict(data_baru)
    
    return final_prediction

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Mendapatkan data JSON dari request
    data = request.get_json()
    
    # Konversi data JSON ke DataFrame
    data_baru = pd.DataFrame([data])

    # Jalankan pipeline prediksi
    hasil_prediksi = predict_pipeline(data_baru)

    # Mengembalikan hasil sebagai JSON
    return jsonify({'prediction': hasil_prediksi[0]})

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
