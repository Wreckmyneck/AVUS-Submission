import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# Download stopwords list (only need to do this once)
nltk.download('stopwords')
nltk.download('punkt')

# Define a function to clean the content
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
human_file_path = r"Datasets\human_merged.csv"
ai_file_path = r"Datasets\generated_merged.csv"

# Read and preprocess data
human_written_text = readfile(human_file_path)[1:]
ai_generated_text = readfile(ai_file_path)[1:]

# Labels (0 for AI-generated text, 1 for human-written text)
labels = [0] * len(ai_generated_text) + [1] * len(human_written_text)

# Combine AI-generated and human-written text
all_text = np.concatenate((ai_generated_text, human_written_text), axis=0)


#Implementation of training and bootstrap method
# Create a pipeline with CountVectorizer, TfidfTransformer, and a Random Forest classifier
ngram_range = (1, 2)  # change the n-gram range here (unigrams and bigrams currently).

#Logistic Regression and SVM were included in initial run of 100 iterations to look at results, the best
#was trained on 1000 iterations and only that due to processing power so the other two models are blaked out
classifiers = [
    #('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    #('SVM', SVC(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)), 
]

# Bootstrap function
def bootstrap(X, y, n_samples=1000):
    models_pipeline = []
    precision = []
    recall = []
    f1 = []
    accuracy = []
    best_f1_score = 0.0
    best_model_pipeline = None
    
    for i in range(n_samples):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        
        model_pipelines = []
        for name, classifier in classifiers:
            pipeline = Pipeline([
                ('vectorizer', CountVectorizer(ngram_range=ngram_range)),  # Adjust ngram_range as needed
                ('tfidf', TfidfTransformer()),
                ('classifier', classifier),
            ])
            
            pipeline.fit(X_train, y_train)
            model_pipelines.append(pipeline)       
            y_pred = pipeline.predict(X_test)
            precision.append(precision_score(y_test, y_pred))
            recall.append(recall_score(y_test, y_pred))
            f1.append(f1_score(y_test, y_pred))
            accuracy.append(accuracy_score(y_test, y_pred))
            current_f1_score = f1_score(y_test, y_pred)  
            if current_f1_score > best_f1_score:
                best_f1_score = current_f1_score
                best_model_pipeline = pipeline        
        models_pipeline.append(model_pipelines)
        print(f"Iteration {i+1}/{n_samples} completed.")   
    models = [name for name, _ in classifiers] * n_samples
    pred_df = pd.DataFrame(
        {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Models": models,
        }
    )
    return models_pipeline, pred_df, best_model_pipeline

# Call the bootstrap function
models_pipeline, pred_df, best_model_pipeline = bootstrap(all_text, labels, n_samples=300)

joblib.dump(best_model_pipeline, 'Random Forest_tfidfngram_pipeline.joblib')

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
boxplot.figure.savefig("performance_boxplot_for_TFIDFNGRAM.png")

# Close the figure to release resources (optional)
plt.close(boxplot.figure)