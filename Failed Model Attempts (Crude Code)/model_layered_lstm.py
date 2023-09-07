"""
This code turned out to not give sufficient results to be further attempted/improved upon within the time-frame.
Due to the time-frame the code is also unrefined and lacking comment.
"""
import numpy as np
import pandas as pd
import csv
import nltk
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GRU

# Function to clean the content
def clean_content(text):
    if text is None:
        text = ''
    # Lowercase conversion
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join the filtered tokens back into a sentence
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

def readfile(filepath):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filepath, sep=",,", encoding="utf-8", quoting=csv.QUOTE_ALL, header=None, names=['index', 'content'], engine='python')
    df['content'] = df['content'].apply(clean_content)
    content_column = df['content'].values
    return content_column

file_path_ai = r"C:\Users\Conor\Desktop\Summer Code\textfiles\Merged\generated_merged.csv"
ai_generated_text = readfile(file_path_ai)  # List of ai-written text samples
ai_generated_text = ai_generated_text[1:]

file_path_human = r"C:\Users\Conor\Desktop\Summer Code\textfiles\Merged\human_merged.csv"
human_written_text = readfile(file_path_human)  # List of human-generated text samples
human_written_text = human_written_text[1:]

labels = [0] * len(ai_generated_text) + [1] * len(human_written_text)
texts = np.concatenate((ai_generated_text, human_written_text), axis=0)

# Tokenization and padding
max_num_words = 10000
tokenizer = Tokenizer(num_words=max_num_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

max_sequence_length = 100  # You can adjust this to your desired sequence length.
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding="post", truncating="post")

# Convert labels to a numpy array
labels = np.array(labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Build the LSTM model
embedding_dim = 100
model = Sequential()
model.add(Embedding(max_num_words, embedding_dim, input_length=max_sequence_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
optimizer = Adam(learning_rate=1e-4)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the model
batch_size = 32
epochs = 10
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
model.save("lstm_model.keras")