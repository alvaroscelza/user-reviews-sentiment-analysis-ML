import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataLoader:
    def __init__(self, file_path, max_words=1000, max_sequence_length=100):
        self.file_path = file_path
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length

    def load_data(self):
        reviews = pd.read_csv(self.file_path)
        scores = reviews["score"].apply(np.ceil)
        # Combine translation_title and translation_text, handling empty translation_text
        texts = reviews.apply(lambda row: self.combine_texts(row['translation_title'], row['translation_text']), axis=1)
        reviews.drop(columns=['translation_title', 'translation_text'], inplace=True)
        texts_deduplicated = texts.drop_duplicates(keep='first')
        scores_deduplicated = scores[texts_deduplicated.index]
        return texts_deduplicated, scores_deduplicated

    @staticmethod
    def combine_texts(title, text):
        if pd.isna(title) and pd.isna(text):
            return ''
        elif pd.isna(text) or text.strip() == '':
            return title
        elif pd.isna(title) or title.strip() == '':
            return text
        else:
            return f"{title} {text}"

    def preprocess_data(self, texts, scores):
        label_encoder = LabelEncoder()
        scores = label_encoder.fit_transform(scores)
        training_texts, testing_texts, training_scores, testing_scores = train_test_split(texts, scores, test_size=0.2, random_state=42)
        tokenizer = Tokenizer(num_words=self.max_words)
        tokenizer.fit_on_texts(training_texts)
        training_tokens = tokenizer.texts_to_sequences(training_texts)
        testing_tokens = tokenizer.texts_to_sequences(testing_texts)
        training_tokens = pad_sequences(training_tokens, maxlen=self.max_sequence_length)
        testing_tokens = pad_sequences(testing_tokens, maxlen=self.max_sequence_length)
        return training_tokens, testing_tokens, training_scores, testing_scores, tokenizer
