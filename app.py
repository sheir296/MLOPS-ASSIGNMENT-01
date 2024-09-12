from flask import Flask, request, jsonify, send_from_directory
from main import predict

app = Flask(__name__, static_folder='frontend', static_url_path='')

@app.route('/')
def home():
    return send_from_directory('frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    data = request.get_json()
    area = data['area']
    bedrooms = data['bedrooms']
    bathrooms = data['bathrooms']
    
    # Get prediction
    prediction = predict(area, bedrooms, bathrooms)
    
    return jsonify({'predicted_price': prediction})

if __name__ == '__main__':
    app.run(debug=True)
