import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# 1. Load data
df = pd.read_csv("data/reviews.csv")

# 2. Basic cleaning
df['review'] = df['review'].str.lower()

# 3. Convert label to numeric
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# 5. Vectorize text
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train ML model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 7. Evaluate
preds = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, preds))

# 8. Save model & vectorizer
pickle.dump(model, open("models/sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("Model trained & saved!")
