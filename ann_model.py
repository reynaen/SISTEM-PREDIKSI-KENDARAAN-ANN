import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("jumlah_kendaraan.csv")

df = df[['tahun', 'jumlah_kendaraan']]
df = df.groupby('tahun').sum().reset_index()
df.columns = ['Tahun', 'Kendaraan']

print(df.head())

# =========================
# VISUALISASI DATA
# =========================
plt.figure()
sns.lineplot(x=df["Tahun"], y=df["Kendaraan"], marker='o')
plt.title("Data Jumlah Kendaraan per Tahun")
plt.xlabel("Tahun")
plt.ylabel("Jumlah Kendaraan")
plt.show()

# =========================
# PREPROCESSING
# =========================
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

X = df_scaled[:, 0].reshape(-1, 1)
Y = df_scaled[:, 1]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# =========================
# MODEL ANN
# =========================
model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, Y_train, epochs=200)

# =========================
# GRAFIK LOSS
# =========================
plt.figure()
plt.plot(history.history['loss'])
plt.title("Grafik Loss Training ANN")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# =========================
# GRAFIK PREDIKSI vs AKTUAL
# =========================
Y_pred = model.predict(X_test)

plt.figure()
plt.scatter(X_test, Y_test)
plt.scatter(X_test, Y_pred)
plt.title("Perbandingan Data Aktual vs Prediksi")
plt.xlabel("Tahun (scaled)")
plt.ylabel("Kendaraan (scaled)")
plt.show()

# =========================
# SAVE MODEL
# =========================
model.save("model.h5")
joblib.dump(scaler, "scaler.save")

print("✅ Model berhasil disimpan!")