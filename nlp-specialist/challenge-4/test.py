import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler, autocast

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class IntentClassifier:
    def __init__(self, model_name='xlm-roberta-base', max_length=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        self.model = None
        self.scaler = GradScaler()

    def prepare_data(self, data):
        texts = [item['query'] for item in data]
        intents = [item['intent'] for item in data]
        
        self.label_encoder.fit(intents)
        labels = self.label_encoder.transform(intents)

        # Initialize the model with the correct number of labels
        num_labels = len(self.label_encoder.classes_)
        self.model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=num_labels)
        self.model.to(self.device)

        return train_test_split(texts, labels, test_size=0.2, random_state=42)

    def train(self, train_texts, train_labels, batch_size=4, num_epochs=5):
        train_dataset = IntentDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        for epoch in range(num_epochs):
            self.model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                if torch.max(labels) >= len(self.label_encoder.classes_):
                    raise ValueError(f"Label index {torch.max(labels).item()} is out of bounds. "
                                     f"Number of classes: {len(self.label_encoder.classes_)}")

                with autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                # Clear CUDA cache
                torch.cuda.empty_cache()

            print(f"Epoch {epoch + 1}/{num_epochs} completed")

    def predict(self, query):
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            with autocast():
                outputs = self.model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
        predicted_intent = self.label_encoder.inverse_transform([predicted.item()])[0]

        top_3 = torch.topk(probabilities, 3, dim=1)
        top_3_intents = self.label_encoder.inverse_transform(top_3.indices[0].tolist())
        top_3_probs = top_3.values[0].tolist()

        return {
            'predicted_intent': predicted_intent,
            'confidence': confidence.item(),
            'top_3_intents': list(zip(top_3_intents, top_3_probs))
        }

    def few_shot_learning(self, new_data):
        new_texts = [item['query'] for item in new_data]
        new_intents = [item['intent'] for item in new_data]

        # Update label encoder with new intents
        self.label_encoder.fit(list(self.label_encoder.classes_) + list(set(new_intents)))

        # Reinitialize the model with the updated number of labels
        num_labels = len(self.label_encoder.classes_)
        self.model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=num_labels)
        self.model.to(self.device)

        # Prepare new data
        new_labels = self.label_encoder.transform(new_intents)
        new_dataset = IntentDataset(new_texts, new_labels, self.tokenizer, self.max_length)
        new_dataloader = DataLoader(new_dataset, batch_size=4, shuffle=True)

        # Fine-tune the model on new data
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        num_epochs = 3

        for epoch in range(num_epochs):
            self.model.train()
            for batch in new_dataloader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                with autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                # Clear CUDA cache
                torch.cuda.empty_cache()

            print(f"Few-shot learning: Epoch {epoch + 1}/{num_epochs} completed")

# Usage example
def main():
    # Load data
    with open('product_review.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize and train the model
    classifier = IntentClassifier()
    train_texts, test_texts, train_labels, test_labels = classifier.prepare_data(data)
    classifier.train(train_texts, train_labels)

    # Test the model
    test_query = "আমার ব্যালেন্স কত?"
    result = classifier.predict(test_query)
    print(f"Query: {test_query}")
    print(f"Predicted Intent: {result['predicted_intent']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("Top 3 Intents:")
    for intent, prob in result['top_3_intents']:
        print(f"  {intent}: {prob:.2f}")

    # Demonstrate few-shot learning
    new_data = [
        {"query": "How do I update my KYC?", "intent": "update_kyc"},
        {"query": "আমার কেওয়াইসি আপডেট করব কিভাবে?", "intent": "update_kyc"}
    ]
    classifier.few_shot_learning(new_data)

    # Test with a query for the new intent
    new_query = "I need to update my KYC information"
    new_result = classifier.predict(new_query)
    print(f"\nQuery: {new_query}")
    print(f"Predicted Intent: {new_result['predicted_intent']}")
    print(f"Confidence: {new_result['confidence']:.2f}")
    print("Top 3 Intents:")
    for intent, prob in new_result['top_3_intents']:
        print(f"  {intent}: {prob:.2f}")

if __name__ == "__main__":
    main()
