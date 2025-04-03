import pandas as pd
import numpy as np
import jsonlines
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Disable GPU warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_diplomacy_data(train_file, test_file, val_file):
    """
    Load Diplomacy game dialog data from jsonlines files
    """

    def process_file(file_path):
        data = []
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                for msg, sender_label in zip(obj['messages'], obj['sender_labels']):
                    data.append({
                        'message': msg,
                        'label': int(sender_label),
                        'speaker': obj['speakers'][0],
                        'receiver': obj['receivers'][0],
                        'game_id': obj['game_id']
                    })
        return pd.DataFrame(data)

    train_df = process_file(train_file)
    test_df = process_file(test_file)
    val_df = process_file(val_file)

    return train_df, test_df, val_df


def preprocess_text(text):
    """
    Clean and preprocess text data
    """
    # Convert to lowercase
    text = str(text).lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def prepare_model_data(dfs, max_words=5000, max_len=150):
    """
    Prepare text data for deep learning model
    """
    # Combine all dataframes
    combined_df = pd.concat(dfs)

    # Preprocess text
    combined_df['processed_text'] = combined_df['message'].apply(preprocess_text)

    # Tokenization
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(combined_df['processed_text'])

    # Prepare datasets
    datasets = []
    for df in dfs:
        # Preprocess
        df['processed_text'] = df['message'].apply(preprocess_text)

        # Convert to sequences
        X = tokenizer.texts_to_sequences(df['processed_text'])
        X = pad_sequences(X, maxlen=max_len)

        # Encode labels
        y = df['label'].astype(int)

        datasets.append((X, y))

    return datasets, tokenizer


def build_enhanced_lstm_model(max_words, max_len):
    """
    Build enhanced LSTM model for deception detection
    """
    model = Sequential([
        # Embedding layer with increased dimensionality
        Embedding(max_words, 64, input_length=max_len),

        # Bidirectional LSTM for capturing context from both directions
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),

        # Another bidirectional LSTM layer
        Bidirectional(LSTM(64)),
        Dropout(0.3),

        # More dense layers for complex feature extraction
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),

        # Output layer
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    return model


def train_and_evaluate(train_X, train_y, test_X, test_y, val_X, val_y):
    """
    Train and evaluate the deception detection model
    """
    # Model parameters
    max_words = 5000
    max_len = 150

    # Build model
    model = build_enhanced_lstm_model(max_words, max_len)

    # Train model
    history = model.fit(
        train_X, train_y,
        epochs=5,
        batch_size=32,
        validation_data=(val_X, val_y),
        verbose=1
    )

    # Evaluate on test set
    y_pred = (model.predict(test_X) > 0.5).astype(int)

    # Detailed evaluation
    print("\nTest Set Evaluation:")
    print("Accuracy:", accuracy_score(test_y, y_pred))
    print("\nClassification Report:")
    print(classification_report(test_y, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(test_y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Sample predictions with details
    print("\nSample Predictions:")
    # Select 20 random indices
    sample_indices = np.random.choice(len(test_X), 20, replace=False)

    for idx in sample_indices:
        # Get raw text
        raw_text = test_X[idx]
        true_label = test_y[idx]
        pred_prob = model.predict(raw_text.reshape(1, -1))[0][0]
        pred_label = 1 if pred_prob > 0.5 else 0

        print(f"\nMessage: {raw_text}")
        print(f"True Label: {true_label}")
        print(f"Predicted Label: {pred_label}")
        print(f"Prediction Probability: {pred_prob:.4f}")
        print("-" * 50)

    return model, history


def main():
    # File paths
    train_file = 'train.jsonl'
    test_file = 'test.jsonl'
    val_file = 'validation.jsonl'

    # Load data
    train_df, test_df, val_df = load_diplomacy_data(train_file, test_file, val_file)

    # Prepare data
    [(train_X, train_y), (test_X, test_y), (val_X, val_y)], tokenizer = prepare_model_data(
        [train_df, test_df, val_df]
    )

    # Train and evaluate
    model, history = train_and_evaluate(train_X, train_y, test_X, test_y, val_X, val_y)


if __name__ == '__main__':
    main()