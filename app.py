from fastapi import FastAPI
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

app = FastAPI()

# Load your trained model
model = DistilBertForSequenceClassification.from_pretrained("model")
tokenizer = DistilBertTokenizerFast.from_pretrained("model")

@app.post("/predict")
async def predict(data: dict):
    text = data["review"]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()

    sentiment = "positive" if pred == 1 else "negative"

    return {
        "sentiment": sentiment,
        "confidence": confidence
    }
