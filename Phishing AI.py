# phishing_detector.py

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset (or replace this with your own dataset path)
# You can download from: https://www.kaggle.com/datasets/sid321axn/phishing-website-detector
data = pd.read_csv("phishing_site_urls.csv")

# --- Step 1: Feature Engineering ---
def extract_features(url):
    return {
        'url_length': len(url),
        'has_ip': int(bool(re.search(r'\d{1,3}(?:\.\d{1,3}){3}', url))),
        'has_https': int('https' in url),
        'has_at_symbol': int('@' in url),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'is_shortened': int(bool(re.search(r"bit\.ly|goo\.gl|tinyurl\.com|ow\.ly", url)))
    }

feature_list = [extract_features(url) for url in data['url']]
features_df = pd.DataFrame(feature_list)

# Label encoding: 'bad' = 1 (phishing), 'good' = 0 (legit)
labels = data['label'].apply(lambda x: 1 if x == 'bad' else 0)

# --- Step 2: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.3, random_state=42)

# --- Step 3: Model Training ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 4: Evaluation ---
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Step 5: Save Model ---
joblib.dump(model, "phishAI_model.pkl")

# --- Step 6: Predict on New URL ---
def predict_url(url):
    features = extract_features(url)
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    return "Phishing" if prediction == 1 else "Legitimate"

# Sample prediction
sample_url = "http://192.168.0.1/verify/account"
print(f"\nSample URL Prediction for '{sample_url}':", predict_url(sample_url))
