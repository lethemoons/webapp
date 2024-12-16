from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import joblib
import numpy as np
from utils import preprocess_input, categorize_bodyfat
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Memuat model
model = joblib.load("saved_model/stacked_model.pkl")

# Route halaman utama
@app.route("/")
def home():
    return render_template("index.html")

# Route untuk halaman input data prediksi
@app.route("/check", methods=["GET"])
def check():
    return render_template("check.html")

# Route untuk menampilkan hasil prediksi
@app.route("/result", methods=["POST"])
def result():
    print(request.form)
    try:
        # Ambil data dari form
        data = {
            "Age": request.form.get("Age"),
            "Weight": request.form.get("Weight"),
            "Height": request.form.get("Height"),
            "Neck": request.form.get("Neck"),
            "Chest": request.form.get("Chest"),
            "Abdomen": request.form.get("Abdomen"),
            "Hip": request.form.get("Hip"),
            "Thigh": request.form.get("Thigh"),
            "Knee": request.form.get("Knee"),
            "Wrist": request.form.get("Wrist"),
            "Sex": request.form.get("Sex")
        }

        # Validasi data input
        for key, value in data.items():
            if value is None or value.strip() == "":
                return f"Error: {key} is required.", 400

        # Konversi data ke tipe yang sesuai
        data = {
            "Age": int(data["Age"]),
            "Weight": float(data["Weight"]),
            "Height": float(data["Height"]),
            "Neck": float(data["Neck"]),
            "Chest": float(data["Chest"]),
            "Abdomen": float(data["Abdomen"]),
            "Hip": float(data["Hip"]),
            "Thigh": float(data["Thigh"]),
            "Knee": float(data["Knee"]),
            "Wrist": float(data["Wrist"]),
            "Sex": data["Sex"].lower()
        }

        # Proses data untuk prediksi
        processed_data = preprocess_input(data)
        bodyfat_prediction = model.predict([processed_data])[0]
        category = categorize_bodyfat(bodyfat_prediction, 1 if data["Sex"] == "male" else 0)
        recommendation = get_recommendations(category, 1 if data["Sex"] == "male" else 0)

        # Kirim data ke template hasil
        return render_template(
            "result.html",
            bodyfat_percentage=round(bodyfat_prediction, 2),
            category=category,
            recommendation=recommendation
        )
    except Exception as e:
        return f"Error processing data: {str(e)}", 500
    
@app.route("/dashboard")
def dashboard():
    return "Ini adalah halaman dashboard."


# Fungsi untuk memberikan rekomendasi gaya hidup berdasarkan kategori dan jenis kelamin
def get_recommendations(category, sex):
    recommendations = {
        "Essential fat": {
            1: (
                "Nutrisi: Konsumsi makanan tinggi lemak sehat seperti omega-3 untuk fungsi hormonal.\n"
                "Olahraga: Latihan intensitas rendah hingga sedang seperti yoga atau jalan kaki untuk menjaga energi.\n"
                "Hidrasi: Pastikan asupan cairan cukup untuk metabolisme."
            ),
            0: (
                "Nutrisi: Konsumsi makanan tinggi lemak sehat seperti omega-3 untuk fungsi hormonal.\n"
                "Olahraga: Latihan intensitas rendah hingga sedang seperti yoga atau jalan kaki untuk menjaga energi.\n"
                "Hidrasi: Pastikan asupan cairan cukup untuk metabolisme."
            )
        },
        "Athletes": {
            1: (
                "Nutrisi: Tingkatkan asupan protein (1.2-2.0 g/kg berat badan) untuk pemulihan.\n"
                "Olahraga: Fokus pada latihan kekuatan (resistance training) dan latihan HIIT.\n"
                "Recovery: Prioritaskan tidur 7-9 jam per malam dan peregangan untuk menghindari cedera."
            ),
            0: (
                "Nutrisi: Tingkatkan asupan protein (1.2-2.0 g/kg berat badan) untuk pemulihan.\n"
                "Olahraga: Fokus pada latihan kekuatan (resistance training) dan latihan HIIT.\n"
                "Recovery: Prioritaskan tidur 7-9 jam per malam dan peregangan untuk menghindari cedera."
            )
        },
        "Fitness enthusiasts": {
            1: (
                "Nutrisi: Konsumsi makanan seimbang (karbohidrat kompleks, protein, lemak sehat).\n"
                "Olahraga: Gabungkan latihan aerobik dan latihan kekuatan minimal 150 menit per minggu.\n"
                "Hidrasi: Hindari minuman bergula; fokus pada air putih dan elektrolit alami."
            ),
            0: (
                "Nutrisi: Konsumsi makanan seimbang (karbohidrat kompleks, protein, lemak sehat).\n"
                "Olahraga: Gabungkan latihan aerobik dan latihan kekuatan minimal 150 menit per minggu.\n"
                "Hidrasi: Hindari minuman bergula; fokus pada air putih dan elektrolit alami."
            )
        },
        "Healthy average": {
            1: (
                "Nutrisi: Kurangi asupan gula tambahan, tingkatkan konsumsi serat dan makanan utuh.\n"
                "Olahraga: Aktivitas fisik sedang, seperti berjalan 30 menit sehari atau aktivitas rekreasi.\n"
                "Hidrasi & Manajemen Stres: Minum cukup cairan dan praktikkan teknik relaksasi seperti meditasi."
            ),
            0: (
                "Nutrisi: Kurangi asupan gula tambahan, tingkatkan konsumsi serat dan makanan utuh.\n"
                "Olahraga: Aktivitas fisik sedang, seperti berjalan 30 menit sehari atau aktivitas rekreasi.\n"
                "Hidrasi & Manajemen Stres: Minum cukup cairan dan praktikkan teknik relaksasi seperti meditasi."
            )
        },
        "Dangerously high (obese)": {
            1: (
                "Nutrisi: Fokus pada defisit kalori yang sehat, konsumsi lebih banyak sayuran, dan hindari makanan olahan.\n"
                "Olahraga: Pilih latihan ringan seperti berjalan kaki atau berenang (awalnya). Tingkatkan intensitas secara bertahap.\n"
                "Dukungan Medis: Konsultasi dengan ahli gizi dan dokter jika diperlukan.\n"
                "Pola Tidur: Prioritaskan tidur untuk regulasi hormon metabolik."
            ),
            0: (
                "Nutrisi: Fokus pada defisit kalori yang sehat, konsumsi lebih banyak sayuran, dan hindari makanan olahan.\n"
                "Olahraga: Pilih latihan ringan seperti berjalan kaki atau berenang (awalnya). Tingkatkan intensitas secara bertahap.\n"
                "Dukungan Medis: Konsultasi dengan ahli gizi dan dokter jika diperlukan.\n"
                "Pola Tidur: Prioritaskan tidur untuk regulasi hormon metabolik."
            )
        }
    }
    return recommendations.get(category, {}).get(sex, "No recommendations available")


if __name__ == "__main__":
    app.run(debug=True)

