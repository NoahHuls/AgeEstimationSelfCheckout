from flask import Flask, request, jsonify
from model import AgeEsitimationModel

model = AgeEsitimationModel()

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict_age():

    results = model.predict("./../AgeEstimationSelfCheckout/wwwroot/images")

    return jsonify(message="Prediction:", data=results)

@app.route('/api/models', methods=['GET', 'POST'])
def modelsEndpoint():
    if request.method == 'POST':
        response = request.json
        if "model" not in response:
            return jsonify({"message": "key 'model' is missing", "models": model.getAvilableModels()})
        if model.setActiveModel(response["model"]):
            return jsonify(model.getActiveModel())
    return jsonify(model.getAvilableModels())

if __name__ == '__main__':
    app.run(debug=True)