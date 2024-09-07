import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AdamW, get_scheduler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from .dataset import IntentDataset

class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.transformer.config.num_labels = num_labels
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)

        # Unfreeze all layers for fine-tuning
        for param in self.transformer.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {'loss': loss, 'logits': logits}

class IntentClassifier:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.scaler = GradScaler()

    def prepare_data(self, data):
        texts = [item['query'] for item in data]
        intents = [item['intent'] for item in data]

        self.label_encoder.fit(intents)
        labels = self.label_encoder.transform(intents)

        num_labels = len(self.label_encoder.classes_)
        self.model = CustomModel(model_name=self.config.model_name, num_labels=num_labels)
        self.model.to(self.config.device)

        return train_test_split(texts, labels, test_size=0.2, random_state=42)

    def train(self, train_texts, train_labels):
        train_dataset = IntentDataset(train_texts, train_labels, self.tokenizer, self.config.max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        total_steps = len(train_dataloader) * self.config.num_epochs
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        for epoch in range(self.config.num_epochs):
            self.model.train()
            accumulated_loss = 0
            for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{self.config.num_epochs}'):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)

                if torch.max(labels) >= len(self.label_encoder.classes_):
                    raise ValueError(f"Label index {torch.max(labels).item()} is out of bounds. "
                                     f"Number of classes: {len(self.label_encoder.classes_)}")

                with autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs['loss']

                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()

                if (step + 1) % self.config.accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                    torch.cuda.empty_cache()

            avg_loss = accumulated_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} completed with loss: {avg_loss:.4f}")

            torch.save(self.model.state_dict(), f"model_checkpoint_epoch_{epoch + 1}.pt")

    def evaluate(self, test_texts, test_labels):
        self.model.eval()
        test_dataset = IntentDataset(test_texts, test_labels, self.tokenizer, self.config.max_length)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=1)

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")

    def predict(self, query):
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            max_length=self.config.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(self.config.device)
        attention_mask = encoding['attention_mask'].to(self.config.device)

        with torch.no_grad():
            with autocast():
                outputs = self.model(input_ids, attention_mask=attention_mask)

        logits = outputs['logits']
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

        self.label_encoder.fit(list(self.label_encoder.classes_) + list(set(new_intents)))

        num_labels = len(self.label_encoder.classes_)
        self.model = CustomModel(model_name=self.config.model_name, num_labels=num_labels)
        self.model.to(self.config.device)

        new_labels = self.label_encoder.transform(new_intents)
        new_dataset = IntentDataset(new_texts, new_labels, self.tokenizer, self.config.max_length)
        new_dataloader = DataLoader(new_dataset, batch_size=4, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        total_steps = len(new_dataloader) * 1  # Use fewer epochs for few-shot learning
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        self.model.train()
        accumulated_loss = 0
        for epoch in range(1):  # Use fewer epochs for few-shot learning
            for step, batch in tqdm(enumerate(new_dataloader), total=len(new_dataloader), desc='Few-Shot Learning'):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)

                if torch.max(labels) >= len(self.label_encoder.classes_):
                    raise ValueError(f"Label index {torch.max(labels).item()} is out of bounds. "
                                     f"Number of classes: {len(self.label_encoder.classes_)}")

                with autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs['loss']

                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()

                if (step + 1) % self.config.accumulation_steps == 0 or (step + 1) == len(new_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                    torch.cuda.empty_cache()

            avg_loss = accumulated_loss / len(new_dataloader)
            print(f"Few-Shot Learning Epoch {epoch + 1} completed with loss: {avg_loss:.4f}")

        torch.save(self.model.state_dict(), "few_shot_model_checkpoint.pt")
