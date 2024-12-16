import numpy as np
import pandas as pd

def preprocess_input(data):
    # Konversi jenis kelamin ke numerik
    data["Sex"] = 1 if data["Sex"].lower() == "male" else 0
    if "Sex" not in data or data["Sex"] is None:
        raise ValueError("Sex is required")


    # Hitung BMI
    data["BMI"] = data["Weight"] / (data["Height"] ** 2)

    # Hitung fitur tambahan
    data["y_sr1"] = abs(abs(data["Neck"]) - ((data["Abdomen"] - np.exp(data["Height"])) / 1.1648)) - ((data["Wrist"] * np.sin(data["Sex"])) - (-1.7683))
    data["y_sr2"] = ((data["BMI"] - (np.sin(data["Sex"]) * (data["Hip"] - (data["Abdomen"] / 0.95054))) * np.cos(data["Sex"] / data["Height"])) - (np.sin(data["Wrist"]) + data["Sex"]))

    # Konversi ke DataFrame dengan urutan yang benar
    feature_order = ['Sex', 'Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 
                     'Thigh', 'Knee', 'Wrist', 'BMI', 'y_sr1', 'y_sr2']
    data_frame = pd.DataFrame([data])[feature_order]
    
    return data_frame.values[0]

# Fungsi untuk kategorisasi body fat berdasarkan jenis kelamin
def categorize_bodyfat(bodyfat_percentage, sex):
    if sex == 1:  # Male
        if bodyfat_percentage <= 5:
            return "Essential fat"
        elif 6 <= bodyfat_percentage <= 13:
            return "Athletes"
        elif 14 <= bodyfat_percentage <= 17:
            return "Fitness enthusiasts"
        elif 18 <= bodyfat_percentage <= 24:
            return "Healthy average"
        else:
            return "Dangerously high (obese)"
    else:  # Female
        if bodyfat_percentage <= 13:
            return "Essential fat"
        elif 14 <= bodyfat_percentage <= 20:
            return "Athletes"
        elif 21 <= bodyfat_percentage <= 24:
            return "Fitness enthusiasts"
        elif 25 <= bodyfat_percentage <= 31:
            return "Healthy average"
        else:
            return "Dangerously high (obese)"
