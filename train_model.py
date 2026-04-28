import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Charger les données Iris (intégrées à sklearn)
iris = load_iris()
X = iris.data
y = iris.target

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

# Entraîner un modèle Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sauvegarder le modèle entraîné dans un fichier
joblib.dump(model, 'model.pkl')
print("Modèle sauvegardé sous model.pkl")