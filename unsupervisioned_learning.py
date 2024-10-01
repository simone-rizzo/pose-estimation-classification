import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings 
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
import umap

# Leggi il dataset
df = pd.read_csv("exercise_angles.csv")

# Seleziona solo le colonne numeriche
numeric_columns = df.select_dtypes(include=[np.number])

# Normalizza i dati numerici
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_columns)

# Prendi le etichette dalla colonna "Label"
labels = df['Label']

# Riduzione della dimensionalità con UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(scaled_data)

# Creazione di una palette di colori per le label uniche
unique_labels = labels.unique()
palette = sns.color_palette("hsv", len(unique_labels))
label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

# Scatter plot con i punti colorati in base alle label
plt.figure(figsize=(10, 7))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[label_to_color[label] for label in labels],
    label=labels,
    s=50  # Dimensione dei punti
)

# Aggiunta di una legenda con i nomi delle etichette
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=10) for i in range(len(unique_labels))]
plt.legend(handles, unique_labels, title="Labels", loc="best")

plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the dataset', fontsize=16)
plt.show()
