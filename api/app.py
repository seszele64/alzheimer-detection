# Basic Flask app to serve as the API for Alzheimer's Disease Detection

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Alzheimer's Disease Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    # Here you will process the file and return predictions
    # For now, return a dummy response
    return jsonify({'result': 'Received file: ' + file.filename})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
