import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from model.train import IntentClassifier
from model.config import Config

def main():
    with open('product_review.json', 'r') as f:
        data = json.load(f)

    config = Config()
    classifier = IntentClassifier(config)
    train_texts, test_texts, train_labels, test_labels = classifier.prepare_data(data)
    classifier.train(train_texts, train_labels)
    classifier.evaluate(test_texts, test_labels)

    query = "আমার ব্যালেন্স কত?"
    result = classifier.predict(query)
    print(f"\nQuery: {query}")
    print(f"Predicted Intent: {result['predicted_intent']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("Top 3 Intents:")
    for intent, prob in result['top_3_intents']:
        print(f"  {intent}: {prob:.2f}")

    new_data = [
        {"query": "How do I update my KYC?", "intent": "update_kyc"},
        {"query": "আমার কেওয়াইসি আপডেট করব কিভাবে?", "intent": "update_kyc"}
    ]
    classifier.few_shot_learning(new_data)

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
