import numpy as np
import pandas as pd
import csv
import nltk
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Load dataset
def clean_content(text):
    if text is None:
        text = ''
    text.replace("%%newline%%", "")
    return text

def readfile(file_path):
    df = pd.read_csv(file_path, sep=",,", encoding="utf-8", quoting=csv.QUOTE_ALL, header=None, names=['index', 'content'], engine='python')
    df['content'] = df['content'].str.lower()
    # Keep only the 'content' column
    content_column = df['content'].apply(clean_content).values
    return content_column

# File paths
human_file_path = r"Datasets\human_merged.csv"
ai_file_path = r"Datasets\generated_merged.csv"

# Read and preprocess data
human_written_text = readfile(human_file_path)[1:]
ai_generated_text = readfile(ai_file_path)[1:]

# Labels (0 for AI-generated text, 1 for human-written text)
labels = [0] * len(ai_generated_text) + [1] * len(human_written_text)

# Combine AI-generated and human-written text
all_text = np.concatenate((ai_generated_text, human_written_text), axis=0)

def analyze_text(text_array):
    sentence_lengths = []
    sentence_probabilities = []

    for text in text_array:
        sentences = sent_tokenize(text)  # Tokenize into sentences
        sentence_lengths.extend([len(sent.split()) for sent in sentences])  # Store sentence lengths
        # Create a frequency distribution of sentence lengths
        freq_dist = FreqDist(len(sent.split()) for sent in sentences)
        total_sentences = len(sentences)
        probabilities = {length: count / total_sentences for length, count in freq_dist.items()}
        sentence_probabilities.append(probabilities)

    return sentence_lengths, sentence_probabilities

human_sentence_lengths, human_sentence_probabilities = analyze_text(human_written_text)
ai_sentence_lengths, ai_sentence_probabilities = analyze_text(ai_generated_text)

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

# Define a maximum length for the feature vectors
max_length = max(max(len(probs_dict) for probs_dict in ai_sentence_probabilities),
                 max(len(probs_dict) for probs_dict in human_sentence_probabilities))
print(max_length)

ai_flattened_features = flatten_probabilities(ai_sentence_probabilities, max_length)
human_flattened_features = flatten_probabilities(human_sentence_probabilities, max_length)


all_features = ai_flattened_features + human_flattened_features
print(len(all_features))
print(len(labels))

# Convert the features to a Pandas DataFrame
feature_columns = [f"feature_{i}" for i in range(1, max_length + 1)]
features_df = pd.DataFrame(all_features, columns=feature_columns)

# Combine the features DataFrame with the labels
combined_df = pd.concat([features_df, pd.Series(labels, name="label")], axis=1)

# Remove NaN values from the combined DataFrame
combined_df = combined_df.dropna()

# Split the combined DataFrame back into features and labels
X = combined_df.drop("label", axis=1)
y = combined_df["label"]

def bootstrap(X, y, n_samples=100):
    models = []
    precision = []
    recall = []
    f1 = []
    accuracy = []
    best_f1_score = 0.0
    best_model = None

    for i in range(n_samples):
        print(f"Iteration {i+1}/{n_samples}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        
        #MultinominalNB and SVC were included in initial run of 100 iterations to look at results, the best
        #was trained on 1000 iterations and only that due to processing power so the other two models are blanked out
        classifier_list = [
            #MultinomialNB(), 
            #RandomForestClassifier() 
            SVC(probability=True)
            ]
        
        for classifier in classifier_list:
            model = classifier.fit(X_train, y_train)
            models.append(model)
            
            y_pred = model.predict(X_test)
            
            precision.append(precision_score(y_test, y_pred))
            recall.append(recall_score(y_test, y_pred))
            f1.append(f1_score(y_test, y_pred))
            accuracy.append(accuracy_score(y_test, y_pred))
            current_f1_score = f1_score(y_test, y_pred)

            if current_f1_score > best_f1_score:
                best_f1_score = current_f1_score
                best_model = model
            
    models_name = [#"MultinominalNB", 
        #"RandomForest",
        "SVM"
        ] * n_samples    
        
    pred_df = pd.DataFrame({
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Models": models_name,
    })
    
    return models, pred_df, best_model

models, pred_df, best_model = bootstrap(X, y, n_samples=1000)

joblib.dump(best_model, "burstiness_classification_SVM.joblib")

# Display boxplot
plt.figure(figsize=(10, 6))
pred_df_melted = pd.melt(pred_df, id_vars=["Models"], value_vars=["Accuracy", "Precision", "Recall", "F1"])
boxplot = sns.boxplot(x="Models", y="value", hue="variable", data=pred_df_melted)
plt.title("Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Models")
plt.xticks(rotation=45)
plt.legend(title="Metrics")
# Save the boxplot as an image file (e.g., PNG)
boxplot.figure.savefig("performance_boxplot_for_Burstiness.png")

# Close the figure to release resources (optional)
plt.close(boxplot.figure)


