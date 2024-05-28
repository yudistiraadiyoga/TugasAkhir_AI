import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

filename = 'data.csv'

try:
    data = pd.read_csv(filename)
except FileNotFoundError:
    data = pd.DataFrame({
        'month': [3, 10, 7, 3, 7, 8, 9, 10, 5, 6],
        'day': [5, 5, 5, 6, 6, 6, 7, 7, 1, 2],
        'FFMC': [86.2, 90.6, 92.3, 85.4, 88.1, 91.5, 89.6, 92.3, 84.9, 90.0],
        'DMC': [26.2, 35.4, 28.6, 33.3, 42.6, 29.4, 32.4, 38.4, 29.1, 31.5],
        'DC': [94.3, 145.4, 128.6, 133.3, 142.6, 129.4, 132.4, 138.4, 129.1, 131.5],
        'ISI': [5.1, 10.2, 8.4, 7.1, 9.6, 6.9, 9.3, 10.4, 7.6, 8.8],
        'temp': [8.2, 18.0, 22.8, 15.5, 20.5, 25.3, 21.8, 23.1, 17.0, 19.3],
        'RH': [51, 33, 27, 42, 35, 26, 30, 28, 44, 36],
        'wind': [6.7, 4.9, 5.4, 5.6, 4.7, 5.1, 5.9, 4.8, 6.2, 5.5],
        'rain': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'area': [0, 0.5, 0.4, 0, 0, 0.3, 0.2, 0.7, 0, 0.1]
    })

def get_user_input():
    new_data = {}
    new_data['month'] = int(input("Masukkan bulan: "))
    new_data['day'] = int(input("Masukkan hari: "))
    new_data['FFMC'] = float(input("Masukkan FFMC: "))
    new_data['DMC'] = float(input("Masukkan DMC: "))
    new_data['DC'] = float(input("Masukkan DC: "))
    new_data['ISI'] = float(input("Masukkan ISI: "))
    new_data['temp'] = float(input("Masukkan suhu (temp): "))
    new_data['RH'] = int(input("Masukkan RH: "))
    new_data['wind'] = float(input("Masukkan kecepatan angin (wind): "))
    new_data['rain'] = float(input("Masukkan curah hujan (rain): "))
    return new_data

new_data = get_user_input()
 
# Mengonversi dictionary new_data menjadi DataFrame
new_data_df = pd.DataFrame([new_data])

# Menambahkan data baru ke DataFrame menggunakan pd.concat
data = pd.concat([data, new_data_df], ignore_index=True)

# Menyimpan DataFrame yang diperbarui ke dalam file
data.to_csv(filename, index=False)



# Mengubah kolom 'month' dan 'day' menjadi kategori numerik
data['month'] = data['month'].astype('category').cat.codes
data['day'] = data['day'].astype('category').cat.codes

# Menambahkan label untuk prediksi (misalnya, 0: tidak ada kebakaran, 1: ada kebakaran)
data['label'] = data['area'].apply(lambda x: 1 if x > 0 else 0)

# Memisahkan fitur dan label
X = data.drop(columns=['area', 'label'])
y = data['label']


# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Menggunakan Random Forest Classifier sebagai model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Memprediksi label untuk data pengujian
y_pred = model.predict(X_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Menampilkan matriks kebingungan
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)


# Memprediksi label untuk data pengujian
y_pred = model.predict(X_test)

# Menghitung akurasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Menampilkan matriks kebingungan
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)


# Contoh prediksi baru
new_data_df = pd.DataFrame([new_data])


# Mendapatkan probabilitas prediksi
proba = model.predict_proba(new_data_df)[0][1]  # Probabilitas untuk kelas "kebakaran"
print(f'Prediction: {proba * 100:.2f}% kemungkinan kebakaran')