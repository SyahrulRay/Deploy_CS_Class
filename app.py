import gradio as gr
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import numpy as np

# Load model & tokenizer
model = TFDistilBertForSequenceClassification.from_pretrained("CS_Classification")
tokenizer = DistilBertTokenizer.from_pretrained("CS_Classification")

# Daftar label
labels = ["delivery", "order", "payment", "refund"]
confidence_threshold = 0.7  # ambang batas prediksi minimum

def classify_text(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1)
    
    pred_confidence = np.max(probs)
    pred_class_idx = tf.argmax(probs, axis=1).numpy()[0]

    if pred_confidence >= confidence_threshold:
        predicted_label = labels[pred_class_idx]
    else:
        predicted_label = "unknown"
    
    # Buat dictionary hasil prediksi
    result = {label: float(probs[0][i]) for i, label in enumerate(labels)}
    result["unknown"] = 1.0 - pred_confidence if predicted_label == "unknown" else 0.0

    return result

# Gradio interface
interface = gr.Interface(fn=classify_text,
                         inputs="text",
                         outputs="label",
                         title="Strict DistilBERT Text Classifier",
                         description="Prediksi hanya salah satu dari 4 kelas. Jika tidak yakin, maka dikategorikan sebagai 'unknown'.")

interface.launch()
