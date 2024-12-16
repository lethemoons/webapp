from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import joblib
import numpy as np
from utils import preprocess_input, categorize_bodyfat

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Untuk keamanan sesi login

# Memuat model
model = joblib.load("saved_model/stacked_model.pkl")

# Route halaman utama
@app.route("/")
def home():
    return render_template("home.html")

# Route untuk halaman register
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        # Simpan data pengguna yang didaftarkan
        username = request.form["username"]
        password = request.form["password"]
        # Tambahkan logika penyimpanan user (misalnya ke database)
        return redirect(url_for('login'))
    return render_template("register.html")

# Route untuk halaman login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        # Cek username dan password (gunakan data hardcoded atau database)
        if username == "user" and password == "password":  # Contoh sederhana
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

# Route untuk halaman dashboard
@app.route("/dashboard")
def dashboard():
    if 'logged_in' in session:
        return render_template("dashboard.html")
    return redirect(url_for('login'))

# Route untuk halaman input data prediksi
@app.route("/get_started", methods=["GET", "POST"])
def get_started():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    if request.method == "POST":
        # Mengirim data ke /predict setelah input selesai
        return redirect(url_for('predict'))
    return render_template("get_started.html")

# Route untuk prediksi body fat
@app.route("/predict", methods=["POST"])
def predict():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    data = request.form.to_dict()  # Mendapatkan data dari form
    processed_data = preprocess_input(data)  # Preprocess input data
    bodyfat_prediction = model.predict([processed_data])[0]  # Prediksi
    category = categorize_bodyfat(bodyfat_prediction, int(data["Sex"]))  # Kategorisasi hasil

    return render_template("recommendation.html", 
                           bodyfat_percentage=bodyfat_prediction, 
                           category=category, 
                           recommendations=get_recommendations(category, int(data["Sex"])))

# Fungsi untuk memberikan rekomendasi gaya hidup
def get_recommendations(category, sex):
    recommendations = {
        "Essential fat": {
            1: "Pertahankan konsumsi nutrisi seimbang dan olahraga rutin yang tidak terlalu intens.",
            0: "Pastikan nutrisi cukup untuk kesehatan tubuh, hindari olahraga berlebihan."
        },
        "Athletes": {
            1: "Pertahankan latihan intensitas tinggi dan atur pola makan untuk mendukung kinerja olahraga.",
            0: "Jaga rutinitas latihan intensitas tinggi dengan asupan nutrisi yang cukup."
        },
        "Fitness enthusiasts": {
            1: "Kombinasikan latihan kekuatan dan kardio, dengan diet kaya protein untuk mempertahankan bentuk tubuh.",
            0: "Fokus pada latihan campuran kardio dan kekuatan, dengan asupan protein dan nutrisi yang seimbang."
        },
        "Healthy average": {
            1: "Pertahankan gaya hidup aktif, tingkatkan aktivitas fisik secara bertahap, dan jaga diet seimbang.",
            0: "Tingkatkan aktivitas fisik sehari-hari dengan pola makan sehat yang kaya serat dan vitamin."
        },
        "Dangerously high (obese)": {
            1: "Mulailah dengan program latihan rendah intensitas, hindari makanan tinggi lemak, dan konsultasikan dengan ahli gizi.",
            0: "Pilih latihan ringan seperti berjalan, tingkatkan sayuran dalam diet, dan konsultasikan dengan profesional."
        }
    }
    return recommendations.get(category, {}).get(sex, "No recommendations available")

# Route untuk logout
@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
