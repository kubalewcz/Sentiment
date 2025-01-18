import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./model/herbert")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer/herbert_base_cased")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

    if predicted_class == 0:
        sentiment = "neutralny"
    elif predicted_class == 1:
        sentiment = "negatywny"
    elif predicted_class == 2:
        sentiment = "pozytywny"
    else:
        sentiment = "niezdecydowany"

    return sentiment

def predict_csv(file):
    data = file

    data.drop('target', inplace=True, axis=1)

    counts = {"neutralny": 0,
              "negatywny": 0,
              "pozytywny": 0,
              "niezdecydowany": 0
              }

    for i, j in data[:10].iterrows():
        x = predict_sentiment(j['text'])
        counts[x] += 1

    return counts

if __name__ == "__main__":
    print(predict_csv("dataset/dataset_test_csv/dataset_test.csv"))