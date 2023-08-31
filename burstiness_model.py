import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
import joblib

def analyze_text(text):
    sentence_lengths = []
    sentence_probabilities = []

    
    sentences = sent_tokenize(text)  # Tokenize into sentences
    sentence_lengths.extend([len(sent.split()) for sent in sentences])  # Store sentence lengths

    # Create a frequency distribution of sentence lengths
    freq_dist = FreqDist(len(sent.split()) for sent in sentences)
    total_sentences = len(sentences)
    probabilities = {length: count / total_sentences for length, count in freq_dist.items()}
    sentence_probabilities.append(probabilities)

    return sentence_lengths, sentence_probabilities

# Convert sentence_probabilities to feature vectors
def flatten_probabilities(probabilities_list, max_length):
    flattened_features = []
    for probs_dict in probabilities_list:
        if not probs_dict:  # Check if the dictionary is empty
            feature_vector = [0]  # Handle the case of an empty dictionary
        else:
             feature_vector = [probs_dict.get(length, 0) for length in range(1, max_length + 1)]
        flattened_features.append(feature_vector)
    return flattened_features

def run_model(text):
    # Define a maximum length for the feature vectors
    model_rf = joblib.load(r"trained_classification_models\burstiness_classification_SVM.joblib")
    max_length = 53
    sentence_lengths, sentence_probabilities = analyze_text(text)
    print(sentence_lengths)
    flattened_features = flatten_probabilities(sentence_probabilities, max_length)
    # Convert the features to a Pandas DataFrame
    feature_columns = [f"feature_{i}" for i in range(1, max_length + 1)]
    features_df = pd.DataFrame(flattened_features, columns=feature_columns)
    #remove NaN values
    features_df = features_df.dropna()
    print(features_df)
    rf_prediction = model_rf.predict(features_df)
    rf_probabilities = np.round(model_rf.predict_proba(features_df) * 100, 2)
    if rf_prediction == 1:
        result = "Human Written Text"
    elif rf_prediction == 0:
        result = "AI-generated Text"
    else:
        result = "Error in prediction"
    final_result = [rf_prediction, result, rf_probabilities]
    return final_result

