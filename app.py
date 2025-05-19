from flask import Flask, request, jsonify
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model dan tokenizer
model_path = "CS_Classification"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)

# Label mapping
label_mapping = {
    0: "Delivery",
    1: "Order",
    2: "Payment",
    3: "Refund"
}


CONFIDENCE_THRESHOLD = 0.7

@app.route("/")
def home():
    return "Text Classification API (TensorFlow) is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Tokenisasi
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
        outputs = model(inputs)

        # Ambil prediksi & confidence
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
        max_prob = np.max(probabilities)
        predicted_class = int(np.argmax(probabilities))

        # Cek threshold
        if max_prob < CONFIDENCE_THRESHOLD:
            predicted_label = "Unknown"
        else:
            predicted_label = label_mapping.get(predicted_class, "Unknown")

        return jsonify({
            "text": text,
            "predicted_label": predicted_label,
            "confidence": round(float(max_prob), 4),
            "predicted_class": predicted_class if predicted_label != "Unknown" else None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)