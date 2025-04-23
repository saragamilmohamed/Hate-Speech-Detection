import re
import string
import nltk

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

# Ensure required NLTK resources are downloaded


class CleanTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, text_column='text', return_tokens=False, join_tokens=True):
        self.text_column = text_column
        self.return_tokens = return_tokens
        self.join_tokens = join_tokens
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def fix_encoding(self, text):
        if not isinstance(text, str):
            return ""
        try:
            return text.encode('latin1').decode('utf-8')
        except Exception:
            return text

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""

        text = self.fix_encoding(text)
        text = text.lower()
        text = re.sub(r'@[\w_]+', '', text)  # Remove mentions
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'#', '', text)  # Remove hashtag symbol
         # Convert emojis to text
        text = re.sub(r'\d+', '', text)  # Remove numbers
        #text = contractions.fix(text)
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = re.sub(r'[^\w\s:]', '', text)  # Remove special characters except emoji colons
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text

    def tokenize(self, text):
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # If it's a Series
        if isinstance(X, pd.Series):
            cleaned = X.apply(self.clean_text)
            tokens = cleaned.apply(self.tokenize)

            if self.return_tokens and not self.join_tokens:
                return tokens
            elif self.return_tokens and self.join_tokens:
                return tokens.apply(lambda x: ' '.join(x))
            else:
                return cleaned

        # If it's a DataFrame
        elif isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            cleaned = X_copy[self.text_column].apply(self.clean_text)
            tokens = cleaned.apply(self.tokenize)

            if self.return_tokens and not self.join_tokens:
                X_copy['tokens'] = tokens
                return X_copy
            elif self.return_tokens and self.join_tokens:
                X_copy['clean_text'] = tokens.apply(lambda x: ' '.join(x))
                return X_copy
            else:
                X_copy['clean_text'] = cleaned
                return X_copy

        # If it's a list of strings
        elif isinstance(X, list):
            cleaned = [self.clean_text(text) for text in X]
            tokens = [self.tokenize(text) for text in cleaned]
            if self.return_tokens and not self.join_tokens:
                return tokens
            elif self.return_tokens and self.join_tokens:
                return [' '.join(t) for t in tokens]
            else:
                return cleaned

        else:
            raise ValueError("Input must be a pandas Series, DataFrame, or list of strings.")
