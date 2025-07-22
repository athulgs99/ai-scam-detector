import re
import string
import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

app = Flask(__name__)

# Sample dataset
data = [
    ("You have won $1000! Click here to claim.", 1),
    ("Urgent: Your account has been compromised. Reset now.", 1),
    ("Free access to Netflix. Sign up now!", 1),
    ("Meeting scheduled for tomorrow at 10am.", 0),
    ("Please review the report before Monday.", 0),
    ("Lunch meeting confirmed with client.", 0)
]

texts, labels = zip(*data)

# Train model
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)

print("Model evaluation:")
print(classification_report(y_test, pipeline.predict(X_test)))

# Save model
joblib.dump(pipeline, "scam_detector.pkl")
model = joblib.load("scam_detector.pkl")

# Flask API
@app.route('/predict', methods=['POST'])
def predict():
    content = request.json.get("text", "")
    pred = model.predict([content])[0]
    prob = model.predict_proba([content])[0][1]
    result = {
        "text": content,
        "prediction": "Scam" if pred == 1 else "Safe",
        "confidence": round(prob * 100, 2)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False)
