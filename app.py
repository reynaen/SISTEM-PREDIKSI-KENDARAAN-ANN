from flask import Flask, render_template, request
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model.h5", compile=False)
scaler = joblib.load("scaler.save")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tahun = int(request.form['tahun'])

    # =========================
    # PREDIKSI
    # =========================
    data = np.array([[tahun]])

    data_scaled = scaler.transform(
        np.column_stack((data, np.zeros(len(data))))
    )[:, 0].reshape(-1, 1)

    pred_scaled = model.predict(data_scaled)

    hasil = scaler.inverse_transform(
        np.column_stack((data_scaled[:, 0], pred_scaled))
    )[:, 1]

    # =========================
    # DATA ASLI
    # =========================
    df = pd.read_csv("jumlah_kendaraan.csv")
    df = df[['tahun', 'jumlah_kendaraan']]
    df = df.groupby('tahun').sum().reset_index()

    # =========================
    # GRAFIK (LEBIH SMOOTH)
    # =========================
    plt.figure(figsize=(6,4))

    # garis data asli
    plt.plot(df['tahun'], df['jumlah_kendaraan'],
             marker='o', label="Data Asli")

    # titik prediksi
    plt.scatter(tahun, hasil[0],
                label="Prediksi", s=100)

    # garis bantu ke prediksi
    plt.axvline(x=tahun, linestyle='--', alpha=0.5)

    plt.xlabel("Tahun")
    plt.ylabel("Jumlah Kendaraan")
    plt.title("Prediksi Jumlah Kendaraan")
    plt.legend()

    # =========================
    # CONVERT KE BASE64
    # =========================
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)

    grafik = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template(
        'result.html',
        tahun=tahun,
        hasil=f"{int(hasil[0]):,}",  # format ribuan 🔥
        grafik=grafik
    )

if __name__ == '__main__':
    app.run(debug=True)