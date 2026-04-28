from flask import Flask, request, jsonify
import joblib
import numpy as np

# Crée une instance de l'application Flask
app = Flask(__name__)

# Charge ton modèle Sklearn (assure-toi d'avoir un modèle sauvegardé en tant que 'model.pkl')
model = joblib.load('model.pkl')

@app.route('/')
def home():
 return "API Flask pour prédictions Sklearn"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données envoyées en POST
        data = request.get_json()
        # Assurez-vous que les données sont sous forme de tableau
        features = np.array(data['features']).reshape(1, -1)
        # Faire la prédiction avec le modèle
        prediction = model.predict(features)
        # Retourner la prédiction sous forme de JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
 app.run(debug=False, port=5000) # Démarre le serveur sur le port 5000
