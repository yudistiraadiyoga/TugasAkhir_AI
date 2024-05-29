import pandas as pd
import tkinter as tk
from tkinter import messagebox
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
        # 'area': [0, 0.5, 0.4, 0, 0, 0.3, 0.2, 0.7, 0, 0.1]
    })

data['month'] = data['month'].astype('category').cat.codes
data['day'] = data['day'].astype('category').cat.codes

data['label'] = data['area'].apply(lambda x: 1 if x > 0 else 0)

X = data.drop(columns=['area', 'label'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# conf_matrix = confusion_matrix(y_test, y_pred)
# print('Confusion Matrix:')
# print(conf_matrix)

def predict():
    try:
        new_data = {
            'month': int(month_entry.get()),
            'day': int(day_entry.get()),
            'FFMC': float(FFMC_entry.get()),
            'DMC': float(DMC_entry.get()),
            'DC': float(DC_entry.get()),
            'ISI': float(ISI_entry.get()),
            'temp': float(temp_entry.get()),
            'RH': int(RH_entry.get()),
            'wind': float(wind_entry.get()),
            'rain': float(rain_entry.get())
        }

        new_data_df = pd.DataFrame([new_data])
        new_data_df['month'] = new_data_df['month'].astype('category').cat.codes
        new_data_df['day'] = new_data_df['day'].astype('category').cat.codes

        proba = model.predict_proba(new_data_df)[0][1] 
        result = f'Prediction: {proba * 100:.2f}% kemungkinan kebakaran'

        
        global data
        data = pd.concat([data, new_data_df], ignore_index=True)
        data.to_csv(filename, index=False)

        messagebox.showinfo("Prediction Result", result)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values")

root = tk.Tk()
root.title("Forest Fire Prediction")

month_label = tk.Label(root, text="Month:")
month_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")
month_entry = tk.Entry(root)
month_entry.grid(row=0, column=1, padx=10, pady=5)

day_label = tk.Label(root, text="Day:")
day_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")
day_entry = tk.Entry(root)
day_entry.grid(row=1, column=1, padx=10, pady=5)

FFMC_label = tk.Label(root, text="FFMC:")
FFMC_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")
FFMC_entry = tk.Entry(root)
FFMC_entry.grid(row=2, column=1, padx=10, pady=5)

DMC_label = tk.Label(root, text="DMC:")
DMC_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")
DMC_entry = tk.Entry(root)
DMC_entry.grid(row=3, column=1, padx=10, pady=5)

DC_label = tk.Label(root, text="DC:")
DC_label.grid(row=4, column=0, padx=10, pady=5, sticky="e")
DC_entry = tk.Entry(root)
DC_entry.grid(row=4, column=1, padx=10, pady=5)

ISI_label = tk.Label(root, text="ISI:")
ISI_label.grid(row=5, column=0, padx=10, pady=5, sticky="e")
ISI_entry = tk.Entry(root)
ISI_entry.grid(row=5, column=1, padx=10, pady=5)

temp_label = tk.Label(root, text="Temperature (temp):")
temp_label.grid(row=6, column=0, padx=10, pady=5, sticky="e")
temp_entry = tk.Entry(root)
temp_entry.grid(row=6, column=1, padx=10, pady=5)

RH_label = tk.Label(root, text="Relative Humidity (RH):")
RH_label.grid(row=7, column=0, padx=10, pady=5, sticky="e")
RH_entry = tk.Entry(root)
RH_entry.grid(row=7, column=1, padx=10, pady=5)

wind_label = tk.Label(root, text="Wind Speed (wind):")
wind_label.grid(row=8, column=0, padx=10, pady=5, sticky="e")
wind_entry = tk.Entry(root)
wind_entry.grid(row=8, column=1, padx=10, pady=5)

rain_label = tk.Label(root, text="Rainfall (rain):")
rain_label.grid(row=9, column=0, padx=10, pady=5, sticky="e")
rain_entry = tk.Entry(root)
rain_entry.grid(row=9, column=1, padx=10, pady=5)

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=10, column=0, columnspan=2, pady=10)

root.mainloop()
