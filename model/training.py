from evaluate import load as evaluate_load
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding
import torch
from sklearn.metrics import precision_recall_fscore_support


train_dataset = load_from_disk('/kaggle/working/train')
val_dataset = load_from_disk('/kaggle/working/validation')
test_dataset = load_from_disk('/kaggle/working/testing')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "allegro/herbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
model.to(device)


def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)


tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)


tokenized_train = tokenized_train.rename_column("target", "labels")
tokenized_val = tokenized_val.rename_column("target", "labels")
tokenized_test = tokenized_test.rename_column("target", "labels")

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


training_args = TrainingArguments(
    output_dir="./model/xxx",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.05,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    lr_scheduler_type='cosine',
    report_to=[],
)



def compute_metrics(eval_pred):
    accuracy_metric = evaluate_load("accuracy")

    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)

    predictions = predictions.cpu()
    labels = torch.tensor(labels).cpu()

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        "accuracy": accuracy['accuracy'],
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()

test_results = trainer.evaluate(tokenized_test)
print(test_results)