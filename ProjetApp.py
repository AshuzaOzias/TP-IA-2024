import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
import matplotlib.pyplot as plt

# Chargement des données
@st.cache_data
def load_data():
    url = "C:/Users/MWAMI/ENB2012_data.xlsx"
    data = pd.read_excel(url)
    return data

data = load_data()

# Affichage des données
st.title("Prédiction des Coûts Énergétiques dans les Processus Miniers")
st.write("## Aperçu des données")
st.write(data.head())

# Préparation des données
X = data.iloc[:, :8] # Toutes les colonnes sauf les deux dernières
y = data.iloc[:,8:]   # La colonne de la consommation énergétique
X
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Standardisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Normalisation des données d'entraînement
normalizer = MinMaxScaler()
X_train_normalized = normalizer.fit_transform(X_train_scaled)

# Normalisation des données de test
X_test_normalized = normalizer.transform(X_test_scaled)

# Supposons que vous ayez déjà séparé vos données en X_train, X_test, y_train, y_test

# Normalisation des données
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

st.write("## PRÉDICTION CIBLE 1")
# Assurez-vous que y_train et y_test sont 1D si vous avez une seule cible
# Si vous avez deux colonnes dans y_train et y_test, sélectionnez-en une
# Exemple: si vous voulez la première colonne
y_train_1d = y_train.iloc[:, 0]  # Utilisez .iloc si y_train est un DataFrame pandas
y_test_1d = y_test.iloc[:, 0]    # Utilisez .iloc si y_test est un DataFrame pandas
y

# Entraînement du modèle
model = GradientBoostingRegressor(n_estimators=100, random_state=50)
model.fit(X_train_normalized, y_train_1d)


# Prédiction
y_pred = model.predict(X_test_normalized)

# Erreur quadratique moyenne
mse = mean_squared_error(y_test_1d, y_pred)

# Coefficient de détermination
r2 = r2_score(y_test_1d, y_pred)

st.write("## Performance du modèle")
st.write(f"Mean Squared Error: {mse}")
st.write(f"Coefficient de détermination: {r2}")


# Interface utilisateur pour les prédictions
st.write("## Prédiction de la Consommation Énergétique")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"Valeur pour {col}", value=float(X[col].mean()))

input_df = pd.DataFrame([input_data])
input_df_scaled = scaler.transform(input_df)
input_df_normalized = normalizer.transform(input_df_scaled)
prediction = model.predict(input_df_normalized)

st.write(f"### Consommation Énergétique Prédite: {prediction[0]}")


# Visualisation de l'écart entre les prédictions et les valeurs réelles
st.write("## Visualisation de l'écart entre les prédictions et les valeurs réelles")

# Conversion de DataFrames Pandas vers Numpy
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

# y_pred_np = y_pred.to_numpy() 
# y_test_1d_np = y_test_1d.to_numpy()
y_pred_np = y_pred
y_test_1d_np = y_test_1d

# Visualisation de la première sortie
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test_1d_np, y_pred_np, c='b', marker='o', label='Predictions')
plt.plot([y_test_1d_np.min(), y_test_1d_np.max()], [y_test_1d_np.min(), y_test_1d_np.max()], 'r--', label='Ligne de réf')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')
plt.title('Visualisation de la première sortie')
plt.legend()
plt.grid()
st.pyplot(plt)

st.write("## PRÉDICTION CIBLE 2")

# Assurez-vous que y_train et y_test sont 1D si vous avez une seule cible
# Si vous avez deux colonnes dans y_train et y_test, sélectionnez-en une
# Exemple: si vous voulez la première colonne
y_train_1d2 = y_train.iloc[:, 1]  # Utilisez .iloc si y_train est un DataFrame pandas
y_test_1d2 = y_test.iloc[:, 1]    # Utilisez .iloc si y_test est un DataFrame pandas
y_test_1d2 

# Choix et entraînement du modèle
model = GradientBoostingRegressor(n_estimators=100, random_state=50)
model.fit(X_train_normalized, y_train_1d2)

# Prédiction des targets avec le modèle choisi
y_pred2 = model.predict(X_test_normalized)
# y_pred2.shape


# Erreur quadratique moyenne
mse2 = mean_squared_error(y_test_1d2, y_pred2)

# Coefficient de détermination
r2_2 = r2_score(y_test_1d2, y_pred2)

st.write("## Performance du modèle")
st.write(f"Mean Squared Error: {mse2}")
st.write(f"Coefficient de détermination: {r2_2}")

# Interface utilisateur pour les prédictions
st.write("## Prédiction de la Consommation Énergétique")
input_data2 = {}
for i, col in enumerate(X.columns):
    input_data2[col] = st.number_input(f"Valeur pour {col}", value=float(X[col].mean()), key=f"input_{i}")

input_df2 = pd.DataFrame([input_data2])
input_df2_scaled = scaler.transform(input_df2)
input_df2_normalized = normalizer.transform(input_df2_scaled)
prediction2 = model.predict(input_df2_normalized)

st.write(f"### Consommation Énergétique Prédite: {prediction2[0]}")

# Visualisation de l'écart entre les prédictions et les valeurs réelles
st.write("## Visualisation de l'écart entre les prédictions et les valeurs réelles")

# Visualisation de la deuxième sortie
y_pred2_np = y_pred2
y_test_1d2_np = y_test_1d2

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 2)
plt.scatter(y_test_1d2_np, y_pred2_np, c='g', marker='o', label='Predictions')
plt.plot([y_test_1d2_np.min(), y_test_1d2_np.max()], [y_test_1d2_np.min(), y_test_1d2_np.max()], 'r--', label='Ligne de réf')
plt.xlabel('Feature 2')
plt.ylabel('Predictions')
plt.title('Visualisation de la deuxième sortie')
plt.legend()
plt.grid()
plt.show()
st.pyplot(plt)

plt.tight_layout()
plt.savefig('GBM_plot.png')
plt.close()

# Visualisation de l'image enregistrée
from PIL import Image

# Ouverture du fichier image
img = Image.open('GBM_plot.png')

# Affichage de l'image
img.show()

# Documentation et cohésion
st.write("## Documentation")
st.write("""
### Cohésion
•  [**Cohésion fonctionnelle**](https://www.bing.com/search?form=SKPBOT&q=Coh%C3%A9sion%20fonctionnelle) : Chaque fonction du code assure une seule tâche bien définie.

•  [**Cohésion séquentielle**](https://www.bing.com/search?form=SKPBOT&q=Coh%C3%A9sion%20s%C3%A9quentielle) : Les étapes de préparation des données, d'entraînement du modèle et de prédiction sont clairement séquencées.

•  [**Cohésion de communication**](https://www.bing.com/search?form=SKPBOT&q=Coh%C3%A9sion%20de%20communication) : Toutes les fonctions travaillent sur les mêmes données.


### Couplage
•  [**Faible couplage**](https://www.bing.com/search?form=SKPBOT&q=Faible%20couplage) : Les composants (chargement des données, préparation, entraînement, prédiction) sont indépendants les uns des autres.


### Sources de Documentation
•  [**[UserGuiding](https://userguiding.com/fr/blog/outils-documentation-logiciels)**](https://www.bing.com/search?form=SKPBOT&q=%5BUserGuiding%5D%28https%3A%2F%2Fuserguiding.com%2Ffr%2Fblog%2Foutils-documentation-logiciels%29) : Cet article présente 16 outils de documentation de logiciels, expliquant leur importance et comment ils peuvent aider les développeurs à créer et maintenir une documentation technique de qualité.

•  [**[ESLSCA](https://www.eslsca.fr/blog/liste-des-sources-de-donnees-pour-preparer-un-projet-de-programmation)**](https://www.bing.com/search?form=SKPBOT&q=%5BESLSCA%5D%28https%3A%2F%2Fwww.eslsca.fr%2Fblog%2Fliste-des-sources-de-donnees-pour-preparer-un-projet-de-programmation%29) : Ce guide pratique explique comment choisir et utiliser les bonnes sources de données pour vos projets de programmation, en mettant l'accent sur la qualité et la pertinence des données.

•  [**[ClickUp](https://clickup.com/fr-FR/blog/110939/outils-de-documentation-des-logiciels)**](https://www.bing.com/search?form=SKPBOT&q=%5BClickUp%5D%28https%3A%2F%2Fclickup.com%2Ffr-FR%2Fblog%2F110939%2Foutils-de-documentation-des-logiciels%29) : Un article qui liste les 10 meilleurs outils de documentation logicielle en 2025, aidant à créer, gérer et partager la documentation technique de manière efficace.

""")



