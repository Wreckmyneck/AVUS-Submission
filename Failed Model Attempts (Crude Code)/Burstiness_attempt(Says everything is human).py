"""
This code turned out to not give sufficient results to be further attempted/improved upon within the time-frame.
Due to the time-frame the code is also unrefined and lacking comment.
"""

from collections import Counter
import pandas as pd
import csv
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Download stopwords list (only need to do this once)
nltk.download('stopwords')
nltk.download('punkt')

def clean_content(text):
    if text is None:
        text = ''
    # Lowercase conversion
    text = text.lower()
    text.replace("%%newline%%", "")
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join the filtered tokens back into a sentence
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

def readfile(file_path):
    df = pd.read_csv(file_path, sep=",,", encoding="utf-8", quoting=csv.QUOTE_ALL, header=None, names=['index', 'content'], engine='python')
    df['content'] = df['content'].str.lower()
    # Keep only the 'content' column
    content_column = df['content'].apply(clean_content).values
    return content_column

# File paths
human_file_path = r"C:\Users\Conor\Desktop\Summer Code\textfiles\Merged\human_merged.csv"
ai_file_path = r"C:\Users\Conor\Desktop\Summer Code\textfiles\Merged\generated_merged.csv"

# Read and preprocess data
human_written_text = readfile(human_file_path)[1:]
ai_generated_text = readfile(ai_file_path)[1:]

# Labels (0 for AI-generated text, 1 for human-written text)
labels = [0] * len(ai_generated_text) + [1] * len(human_written_text)
# Combine AI-generated and human-written text
all_text = np.concatenate((ai_generated_text, human_written_text), axis=0)

def calculate_burstiness(texts, window_size):
    word_counters = [Counter(text.split()) for text in texts]
    burstiness_scores = []

    for word_counter in word_counters:
        N1 = sum(1 for count in word_counter.values() if count == 1)
        E_N1 = np.mean([(1 - np.exp(-count)) / (1 - np.exp(-1)) for count in word_counter.values()])
        burstiness = (N1 - E_N1) / (N1 + E_N1)
        burstiness_scores.append(burstiness)

    return burstiness_scores

burstiness_scores_cleaned = []
labels_cleaned = []
burstiness_scores = calculate_burstiness(all_text, window_size=10)  # Adjust window_size as needed
for i in range(len(burstiness_scores)):
    if not np.isnan(burstiness_scores[i]):
        burstiness_scores_cleaned.append(burstiness_scores[i])
        labels_cleaned.append(labels[i])

X = np.array(burstiness_scores_cleaned).reshape(-1, 1)
y = np.array(labels_cleaned)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)