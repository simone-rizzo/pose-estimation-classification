import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Leggi il dataset
df = pd.read_csv("exercise_angles.csv")

# Separa la colonna delle etichette (Label)
labels = df['Label']

# Seleziona le colonne numeriche
numeric_columns = df.select_dtypes(include=[np.number])

# Seleziona le colonne categoriche
categorical_columns = df.select_dtypes(include=['object']).drop(columns=['Label'])

# Trasforma le colonne categoriche con get_dummies
categorical_data = pd.get_dummies(categorical_columns)

# Normalizza le colonne numeriche
scaler = StandardScaler()
numeric_data_scaled = scaler.fit_transform(numeric_columns)

# Codifica le etichette con LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Converte le etichette in formato categorico
categorical_labels = to_categorical(encoded_labels)

# Combina i dati numerici e categorici
processed_data = np.hstack([numeric_data_scaled, categorical_data])

# Dividi il dataset in training, validation e test (70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(processed_data, categorical_labels, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Definisci il modello di rete neurale
model = Sequential()
model.add(Dense(64, input_shape=(processed_data.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(categorical_labels.shape[1], activation='softmax'))  # Output layer

# Compila il modello
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Addestra il modello
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=32)

# Valuta il modello sul test set (una volta)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Stampa i risultati
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Grafico della perdita di addestramento e validazione
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Inferenza sul test set (usiamo la stessa valutazione fatta prima)
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Stampa qualche predizione rispetto alle etichette vere
for i in range(10):
    print(f"Predicted: {label_encoder.inverse_transform([predicted_labels[i]])[0]}, True: {label_encoder.inverse_transform([true_labels[i]])[0]}")
