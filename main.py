import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./model/herbert")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer/herbert_base_cased")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

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
        sentiment = "ambigous"

    return sentiment


if __name__ == "__main__":
    # Test the function with new sentences
    new_sentences = [
        "Oceniam to miejsce średnio"

    ]

    for sentence in new_sentences:
        sentiment = predict_sentiment(sentence)
        print(f"Sentence: {sentence}\nPredicted Sentiment: {sentiment}\n")