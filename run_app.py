from flask import Flask, jsonify, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('models/model_minimal.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    vectorizer = data['vectorizer']

@app.route('/')
def home():
    return '''
    <html>
    <body>
        <h1>CodeGuard AI - Minimal Version</h1>
        <p>Use /predict?message=your+commit+message</p>
        <p>Example: <a href="/predict?message=Fix+memory+leak">/predict?message=Fix+memory+leak</a></p>
    </body>
    </html>
    '''

@app.route('/predict')
def predict():
    message = request.args.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'})
    
    # Transform and predict
    X = vectorizer.transform([message])
    proba = model.predict_proba(X)[0, 1]
    is_bug = proba > 0.5
    
    return jsonify({
        'message': message,
        'is_bug_fix': bool(is_bug),
        'probability': float(proba),
        'confidence': 'HIGH' if proba > 0.7 or proba < 0.3 else 'MEDIUM'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})

if __name__ == '__main__':
    print("Server starting on http://localhost:5002")
    print("Try: http://localhost:5002/predict?message=Fix+bug")
    app.run(port=5002, debug=True)
