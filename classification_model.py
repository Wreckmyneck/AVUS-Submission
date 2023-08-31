import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

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

def tf_idf_n_gram_model(Text):
    # Load the saved model
    random_forest_pipeline = joblib.load("trained_classification_models\Random Forest_tfidfngram_pipeline.joblib")
    cleaned_text = [clean_content(Text)]
    #predictions using the random forest model:
    random_forest_predictions = random_forest_pipeline.predict(cleaned_text)
    random_forest_probability = np.round(random_forest_pipeline.predict_proba(cleaned_text) * 100, 2)
    return_prediction = []
    if random_forest_predictions == 1:
        return_prediction = [random_forest_predictions, "Human Written Text Detected", random_forest_probability]
    else:
        return_prediction = [random_forest_predictions, "AI generated Text Detected", random_forest_probability]

    return return_prediction

def tfidf_model(Text):
    #load the saved model
    NB_tfidf_only_pipeline = joblib.load("trained_classification_models\MultinomialNB_TFIDF_pipeline.joblib")
    cleaned_text = [clean_content(Text)]

    #predictions using the SVM model:
    NB_tfidf_only_predictions = NB_tfidf_only_pipeline.predict(cleaned_text)
    NB_tfidf_only_probability = np.round(NB_tfidf_only_pipeline.predict_proba(cleaned_text) * 100, 2)
    return_prediction = []
    if NB_tfidf_only_predictions == 1:
        return_prediction = [NB_tfidf_only_predictions, "Human Written Text Detected", NB_tfidf_only_probability]
    else:
        return_prediction = [NB_tfidf_only_predictions, "AI generated Text Detected", NB_tfidf_only_probability]

    return return_prediction
