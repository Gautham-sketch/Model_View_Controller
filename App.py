from flask import Flask, jsonify, request
from Classifier import get_prediction

app = Flask(__name__)

@app.route("/predict", methods = ["POST"])
def predict_data():
    image = request.files.get("alphabet")
    predictor = get_prediction(image)
    return jsonify({
        "prediction" : predictor
    }), 200

if __name__ == '__main__':
    app.run(debug=True)