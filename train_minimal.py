import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

print("Training minimal model...")

# Load data
df = pd.read_csv('data/commits_minimal.csv')

# Simple training
vectorizer = TfidfVectorizer(max_features=50)
X = vectorizer.fit_transform(df['message'])
y = df['is_bug_fix']

model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# Save
import os
os.makedirs('models', exist_ok=True)

with open('models/model_minimal.pkl', 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': vectorizer}, f)

print("Model saved to models/model_minimal.pkl")
print(f"Accuracy: {model.score(X, y):.1%}")
