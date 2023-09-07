import os
import openpyxl
import perplexity_model as model
import classification_model as model2
import numpy as np
import pandas as pd
import csv


def clean_content(text):
    if text is None:
        text = ''
    return text

def readfile(file_path):
    df = pd.read_csv(file_path, sep=",,", encoding="utf-8", quoting=csv.QUOTE_ALL, header=None, names=['index', 'content'], engine='python')
    df['content'] = df['content'].str.lower()
    # Keep only the 'content' column
    content_column = df['content'].apply(clean_content).values
    return content_column

# File paths
human_file_path = r"C:\Users\Conor\Desktop\Summer Code\textfiles\Merged\Test_human.csv"
ai_file_path = r"C:\Users\Conor\Desktop\Summer Code\textfiles\Merged\generatedtestdatafile.csv"

# Read and preprocess data
human_written_text = readfile(human_file_path)[1:]
ai_generated_text = readfile(ai_file_path)[1:]

# Labels (0 for AI-generated text, 1 for human-written text)
labels = [0] * len(ai_generated_text) + [1] * len(human_written_text)
# Combine AI-generated and human-written text
all_text = np.concatenate((ai_generated_text, human_written_text), axis=0)

def run_test():
    # Load the Excel workbook
    workbook = openpyxl.load_workbook("test_results.xlsx")
    worksheet = workbook['Sheet1']   

    index = 0
    for text in all_text:
        print(index)
        average_perplexity, threshold_value, model2_classification, model3_classification, machine_lines, filtered_text = use_model(text)
        result = determine_result(threshold_value, model2_classification, model3_classification)
        cell2 = 0
        if (result == "Human written text detected" and labels[index] == 1):
           # Convert the value to an integer and increment by 1
            cell = 'B3'
            cell_value = worksheet[cell].value
            if cell_value is None:
                new_value = 1
            else:
                new_value = int(cell_value) + 1
           
        elif(result == "Human written text detected" and labels[index] == 0):
           # Convert the value to an integer and increment by 1
            cell = 'B4'
            cell_value = worksheet[cell].value
            if cell_value is None:
                new_value = 1
            else:
                new_value = int(cell_value) + 1
           
            
        if (result == "AI-generated text detected" and labels[index] == 0):
            # Convert the value to an integer and increment by 1
            cell = 'B2'
            cell_value = worksheet[cell].value
            if cell_value is None:
                new_value = 1
            else:
                new_value = int(cell_value) + 1
           
        elif (result == "AI-generated text detected" and labels[index] == 1):
           # Convert the value to an integer and increment by 1
            cell = 'B5'
            cell_value = worksheet[cell].value
            if cell_value is None:
                new_value = 1
            else:
                new_value = int(cell_value) + 1

        if (result == "Human text modified by AI"):
            # Convert the value to an integer and increment by 1
            cell = 'B8'
            cell_value = worksheet[cell].value
            if cell_value is None:
                new_value = 1
            else:
                new_value = int(cell_value) + 1
            
            if(labels[index == 1]):
                cell2 = 'C8'
                cell_value = worksheet[cell2].value
                if cell_value is None:
                    new_value_2 = 1
                else:
                    new_value_2 = int(cell_value) + 1
            elif(labels[index == 0]):
                cell2 = 'D8'
                cell_value = worksheet[cell2].value
                if cell_value is None:
                    new_value_2 = 1
                else:
                    new_value_2 = int(cell_value) + 1

        # Update the cell with the new value
        worksheet[cell] = new_value
        if(cell2 != 0):
            worksheet[cell2] = new_value_2 
        cell2 = 0       
        index = index + 1
    workbook.save('test_results.xlsx')

    # Close the workbook
    workbook.close()

           

def use_model(text):
   perplexity_values, filtered_text = model.split_lines(text)
   average_perplexity = model.average_line_perplexity(perplexity_values)
   threshold_value = model.thresholds(average_perplexity)
   machine_lines = "redudant for tests"
   model2_classification = model2.tf_idf_n_gram_model(text)
   model3_classification = model2.tfidf_multinominal(text)
   return average_perplexity, threshold_value, model2_classification, model3_classification, machine_lines, filtered_text

def determine_result(perplexity_threshold_value, model2_classification, model3_classification):
   """
   Perplexity value is 1 for human, 0.5 for likely AI modified and 0 for AI, classification model 2 and 3 is binary 1 for human 0 for AI
   Add these up means 3, 2 is definitely human, 0, 1 is definitely AI. - 3 means all 3 models voted human, 2 means 2 models voted human and 1 AI, 1 means 1 model voted human 2 voted ai, 0 means all voted AI  
   """
   
   model2_value = model2_classification[0]
   print(model2_value)
   model3_value = model3_classification[0]
   print(model3_value)
   result = perplexity_threshold_value + model2_value + model3_value
   print(result)

   if (result == 3 or result == 2 or result == 2.5):
      text = "Human written text detected"
   elif(result == 1.5):
      text = "Human text modified by AI"
   elif (result == 1 or result == 0 or result == 0.5):
      text = "AI-generated text detected"
   else:
      text = "Error occured please check results below for analysis"
   return text

run_test()