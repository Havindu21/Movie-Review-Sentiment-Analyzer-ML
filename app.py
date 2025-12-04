from fastapi import FastAPI
import pickle

app = FastAPI()

# Load model & vectorizer
model = pickle.load(open("models/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

@app.post("/predict")
def predict(data: dict):
    review = data["review"]
    review_vec = vectorizer.transform([review])
    prediction = model.predict(review_vec)[0]
    sentiment = "positive" if prediction == 1 else "negative"
    return {"sentiment": sentiment}
