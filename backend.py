import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

model = AutoModelForSequenceClassification.from_pretrained("kubalewcz/sentbert")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer/herbert_base_cased")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

    if predicted_class == 0:
        sentiment = "neutral"
    elif predicted_class == 1:
        sentiment = "negative"
    elif predicted_class == 2:
        sentiment = "positive"
    else:
        sentiment = "unknown"

    return sentiment

def predict_csv(file, subset_value):
    data = file


    # counts = {"neutralny": 0,
    #           "negatywny": 0,
    #           "pozytywny": 0,
    #           "niezdecydowany": 0
    #           }

    subset = data.head(subset_value)
    data.loc[subset.index, "Prediction"] = subset["text"].apply(predict_sentiment)

    mapping = {
        0: "neutral",
        1: "negative",
        2: "positive",
        3: "unknown"
    }

    data["target"] = data["target"].map(mapping)

    return data


if __name__ == "__main__":
    print(predict_csv("dataset/dataset_test_csv/dataset_test.csv"))