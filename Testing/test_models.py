import requests
import pandas as pd
import numpy as np
import csv
import openpyxl

# Load dataset
def clean_content(text):
    if text is None:
        text = ''
    # Lowercase conversion

    return text

def readfile(file_path):
    df = pd.read_csv(file_path, sep=",,", encoding="utf-8", quoting=csv.QUOTE_ALL, header=None, names=['index', 'content'], engine='python')
    df['content'] = df['content'].str.lower()
    # Keep only the 'content' column
    content_column = df['content'].apply(clean_content).values
    return content_column

# File paths
human_file_path = r"Datasets\Test_human.csv"
ai_file_path = r"Datasets\generatedtestdatafile.csv"

# Read and preprocess data
human_written_text = readfile(human_file_path)[1:]
ai_generated_text = readfile(ai_file_path)[1:]

# Labels (0 for AI-generated text, 1 for human-written text)
labels = [0] * len(ai_generated_text) + [1] * len(human_written_text)

# Combine AI-generated and human-written text
all_text = np.concatenate((ai_generated_text, human_written_text), axis=0)


    

workbook = openpyxl.load_workbook('Testing/test_results.xlsx')
worksheet = workbook['Sheet1']
def all_results(endpoint, all_text_passed):
    index = 0
    for text in all_text_passed:
        data = {
        "input_data":text,
        "apikey":"securekey"
        }
        response = requests.post(endpoint, json=data)
        if response.status_code == 200:
            result_json = response.json()
            result = result_json["full_results"]
            burstiness_result_json = result.get("burstiness_number")
            burstiness_result = burstiness_result_json[0]
            tfidf_result_json = result.get("model3_number")
            tfidf_result = tfidf_result_json[0]
            tfidfngram_result_json = result.get("model2_number")
            tfidfngram_result = tfidfngram_result_json[0]
            average_perplexity_json = result.get("average_perplexity")
            average_perplexity = average_perplexity_json
            main_result_json = result.get("main_result")
            main_result_string = main_result_json
            if main_result_string == "Human written text detected":
                main_result = 1
            else:
                main_result = 0
            print(average_perplexity)
            if average_perplexity <= 100:
                perplexity_result = 0
            else:
                perplexity_result = 1
            process_results(burstiness_result, tfidf_result, tfidfngram_result, perplexity_result, main_result, index)
            index = index+1
    workbook.close()

def process_results(burstiness_result, tfidf_result, tfidfngram_result, perplexity_result, main_result, index):
    cell_burstiness = 0
    cell_tfidf = 0
    cell_tfidfngram = 0
    cell_perplexity = 0
    cell_main = 0
    cells = []
    print(labels[index])
    print(f"Burstiness: {burstiness_result}")
    print(f"tfidf: {tfidf_result}")
    print(f"tfidfngram: {tfidfngram_result}")
    print(f"perplexity: {perplexity_result}")
    print(f"Main result: {main_result}")
    if labels[index] == 0:
        if burstiness_result == 0:
            cell_burstiness = worksheet['F13'] #Guessed AI right because it is AI
        else:
            cell_burstiness = worksheet['I13'] #Guessed human wrong because it is AI
    else:
        if burstiness_result == 0:
            cell_burstiness = worksheet['H13'] #cell for guessed AI wrong because it is human
        else:
            cell_burstiness= worksheet['G13'] #Cell for guessed human right because it is human
    
    if labels[index] == 0:
        if tfidf_result == 0:
            cell_tfidf = worksheet['A13'] #Guessed AI right because it is AI
        else:
            cell_tfidf = worksheet['D13'] #Guessed human wrong because it is AI
    else:
        if tfidf_result == 0:
            cell_tfidf = worksheet['C13'] #cell for guessed AI wrong because it is human
        else:
            cell_tfidf= worksheet['B13'] #Cell for guessed human right because it is human
    
    if labels[index] == 0:
        if tfidfngram_result == 0:
            cell_tfidfngram  = worksheet['F3'] #Guessed AI right because it is AI
        else:
            cell_tfidfngram  = worksheet['I3'] #Guessed human wrong because it is AI
    else:
        if tfidfngram_result == 0:
            cell_tfidfngram  = worksheet['H3'] #cell for guessed AI wrong because it is human
        else:
            cell_tfidfngram = worksheet['G3'] #Cell for guessed human right because it is human

    if labels[index] == 0:
        if perplexity_result == 0:
            cell_perplexity = worksheet['A3'] #Guessed AI right because it is AI
        else:
            cell_perplexity = worksheet['D3'] #Guessed human wrong because it is AI
    else:
        if perplexity_result == 0:
            cell_perplexity = worksheet['C3'] #cell for guessed AI wrong because it is human
        else:
            cell_perplexity = worksheet['B3'] #Cell for guessed human right because it is human

    if labels[index] == 0:
        if main_result == 0:
            cell_main = worksheet['A22'] #Guessed AI right because it is AI
        else:
            cell_main = worksheet['D22'] #Guessed human wrong because it is AI
    else:
        if main_result == 0:
            cell_main = worksheet['C22'] #cell for guessed AI wrong because it is human
        else:
            cell_main = worksheet['B22'] #Cell for guessed human right because it is human

    cells.append(cell_burstiness)
    cells.append(cell_tfidf)
    cells.append(cell_tfidfngram)
    cells.append(cell_perplexity)
    cells.append(cell_main)

    for cell in cells:
        current_value = cell.value
        new_value = current_value + 1
        cell.value = new_value
    
    workbook.save('Testing/test_results.xlsx')

endpoint_all = r"http://127.0.0.1:5001/all_results"
all_results(endpoint_all, all_text)
    






"""def Burstiness(endpoint, all_text_passed):
    for text in all_text_passed:
        data = {
        "input_data":text,
        "apikey":"securekey"
        }
        response = requests.post(endpoint, json=data)
        if response.status_code == 200:
            result_json = response.json()
            result = result_json["burstiness_classification"]
            binary_result = result.get("classification_number")"""
            