import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os
import numpy as np
import re
import jieba
import tqdm
import csv
from sklearn.model_selection import train_test_split


class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_time_step_output = lstm_out[:, -1, :]
        dropped_output = self.dropout(last_time_step_output)
        output = self.fc(dropped_output)
        return output


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        char_ids = encoded_text['input_ids'].squeeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return char_ids, label


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    seg_list = jieba.cut(text)
    tokenized_text = " ".join(seg_list)
    stop_words_chinese = ["我", "你", "他", "她", "的", "了", "是", "在", "也", "都"]
    tokenized_text = " ".join([word for word in tokenized_text.split() if word not in stop_words_chinese])
    return tokenized_text


def load_data_from_csv(csv_path):
    texts = []
    labels = []
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            texts.append(row["TEXT"])
            labels.append(int(row["label"]))
    return texts, labels


def train_model(model, train_loader, optimizer, criterion, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for char_ids, labels in progress_bar:
            char_ids = char_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(char_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_loss:.4f}")


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for char_ids, labels in tqdm.tqdm(val_loader, desc="Evaluating"):
            char_ids = char_ids.to(device)
            labels = labels.to(device)
            outputs = model(char_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
    accuracy = correct_predictions / total_predictions
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy


def predict_text(model, text, tokenizer, max_len, device):
    model.eval()
    processed_text = preprocess_text(text)
    encoded_text = tokenizer.encode_plus(
        processed_text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    char_ids = encoded_text['input_ids'].to(device)
    with torch.no_grad():
        output = model(char_ids)
        _, prediction = torch.max(output, dim=1)

    label_map = {0: "Not Cyberbullying", 1: "Cyberbullying"}
    return label_map[prediction.item()]


if __name__ == "__main__":
    pretrained_bert_path = "bert-base-chinese"
    max_len = 128
    embedding_dim = 256
    hidden_dim = 128
    num_classes = 2
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 32

    csv_path = "data/Book1.csv"

    print("Loading and preprocessing data...")
    texts, labels = load_data_from_csv(csv_path)

    processed_texts = [preprocess_text(text) for text in texts]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        processed_texts, labels, test_size=0.2, random_state=42
    )

    print("Initializing tokenizer and datasets...")
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert_path)
    vocab_size = tokenizer.vocab_size

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleTextClassifier(vocab_size, embedding_dim, hidden_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("Starting model training...")
    train_model(model, train_loader, optimizer, criterion, device, num_epochs)

    print("\nStarting model evaluation...")
    evaluate_model(model, val_loader, criterion, device)

    print("\n--- Prediction Examples ---")
    test_text_1 = "你今天看起来很不错！"
    prediction_1 = predict_text(model, test_text_1, tokenizer, max_len, device)
    print(f"Input: '{test_text_1}'")
    print(f"Prediction: {prediction_1}\n")

    test_text_2 = "你真是个没用的废物，赶紧滚开。"
    prediction_2 = predict_text(model, test_text_2, tokenizer, max_len, device)
    print(f"Input: '{test_text_2}'")
    print(f"Prediction: {prediction_2}\n")