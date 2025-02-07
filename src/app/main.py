from flask import Flask, request, jsonify, render_template
import joblib
import os
import re
import spacy
import subprocess
import sys

try:
    # Attempt to load the 'en_core_web_lg' model
    nlp = spacy.load("en_core_web_lg")
    print("Model is already installed.")
except OSError:
    print("Model not found. Downloading 'en_core_web_lg'...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_lg"], check=True)
    nlp = spacy.load("en_core_web_lg")
    print("Model installed and loaded.")

app = Flask(__name__)

# Load the model, vectorizer, and label encoder
model_path = os.path.join(os.path.dirname(__file__), 'model')
try:
    model = joblib.load(os.path.join(model_path, 'text_classification_model.pkl'))
    vectorizer = joblib.load(os.path.join(model_path, 'tfidf_vectorizer.pkl'))
    label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
    print("Models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    model, vectorizer, label_encoder = None, None, None

whitespace_re = re.compile(r'\s+')

def text_pre_process(text):
    text = text.lower().strip()
    text = whitespace_re.sub(' ', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_digit]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer or not label_encoder:
        return jsonify({'error': 'Model not loaded correctly. Please check your model files.'}), 500

    data = request.form.get('text')
    if not data:
        return render_template('index.html', prediction="Please enter some text.")

    cleaned_text = text_pre_process(data)
    X_new = vectorizer.transform([cleaned_text])
    predicted_class = model.predict(X_new)
    predicted_label = label_encoder.inverse_transform(predicted_class)

    return render_template('index.html', prediction=f" : {predicted_label[0]}", input_text=data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
