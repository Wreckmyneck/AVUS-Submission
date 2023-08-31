from flask import Flask, request, jsonify
import numpy as np
import perplexity_model as model1
import classification_model as model2
import burstiness_model as burstiness
import requests


app = Flask(__name__)

def use_perplexity_model(text):
   perplexity_values, filtered_text = model1.split_lines(text)
   average_perplexity = model1.average_line_perplexity(perplexity_values)
   threshold_value = model1.thresholds(average_perplexity)
   machine_lines = lines_to_highlight_as_machine_written(perplexity_values)
   return average_perplexity, threshold_value,  machine_lines, filtered_text

def tf_idf_n_gram_model(text):
      model2_classification = model2.tf_idf_n_gram_model(text)
      return model2_classification

def tf_idf_model(text):
   """
    Returns a binary number value 0 for AI, 1 for human in index - 0, and in index 1 is a text result "Human writing detected" or "AI generated text detected"
   """
   model3_classification = model2.tfidf_model(text)
   return model3_classification

def burstiness_model(text):
    burstiness_model = burstiness.run_model(text)
    return burstiness_model

def lines_to_highlight_as_machine_written(values):
   machine_written_lines = []
   for index, value in enumerate(values):
      if(value <= 90):
         machine_written_lines.append(index)
   return machine_written_lines


def determine_result(perplexity_threshold_value, model2_classification, model3_classification, burstiness_classification):
   """
   Perplexity value is 1 for human, 0.5 for likely AI modified and 0 for AI, classification model 2 and 3 is binary 1 for human 0 for AI
   Add these up means 3, 2 is definitely human, 0, 1 is definitely AI. - 3 means all 3 models voted human, 2 means 2 models voted human and 1 AI, 1 means 1 model voted human 2 voted ai, 0 means all voted AI  
   """
   
   model2_value = model2_classification[0]
   model3_value = model3_classification[0]
   burstiness_value = burstiness_classification[0]
   result = perplexity_threshold_value + model2_value + model3_value + burstiness_value

   if (result == 4 or result == 3.5 or result == 3 or result == 2.5):
      text = "Human written text detected"
   elif(result == 2):
      text = "Human text modified by AI"
   elif (result == 1.5 or result == 1 or result == 0.5 or result == 0):
      text = "AI-generated text detected"
   else:
      text = "Error occured please check results below for analysis"
   return text

@app.route('/all_results', methods=['POST'])
def all_results_endpoint():
    try:
        data = request.get_json()
        input_text = data.get('input_data')
        apikey = data.get('apikey')
        if(checkAPIkey(apikey) == True):
            if input_text is None or len(input_text) < 250:
                return jsonify({"error": " 'text' parameter missing or too short"}), 400
            else:
                average_perplexity, threshold_value,  machine_lines, filtered_text = use_perplexity_model(input_text)
                model2_all = tf_idf_n_gram_model(input_text)
                model2_number = model2_all[0].tolist()
                model2_text = model2_all[1]
                model2_probability = model2_all[2].tolist()
                model3_all = tf_idf_model(input_text)
                model3_number = model3_all[0].tolist()
                model3_text = model3_all[1]
                model3_probability = model3_all[2].tolist()
                burstiness_all = burstiness_model(input_text)
                burstiness_number = burstiness_all[0].tolist()
                burstiness_text = burstiness_all[1]
                burstiness_probability = burstiness_all[2].tolist()
                print(burstiness_text)
                main_result = determine_result(threshold_value, model2_all, model3_all, burstiness_all)
                result = {
                "average_perplexity": average_perplexity,
                "threshold_value": threshold_value,
                "model2_number": model2_number,
                "model2_text": model2_text,
                "model2_probability":model2_probability,
                "model3_number": model3_number,
                "model3_text": model3_text,
                "model3_probability":model3_probability,
                "machine_lines": machine_lines,
                "filtered_text": filtered_text,
                "burstiness_number":burstiness_number,
                "burstiness_text":burstiness_text,
                "burstiness_probability":burstiness_probability,
                "main_result": main_result
            }
            return jsonify({"full_results": result}), 200
        else:
            return jsonify({"error": "access denied"}), 500
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route('/perplexity', methods=['POST'])
def perplexity_endpoint():
    try:
        data = request.get_json()
        input_text = data.get('input_data')
        apikey = data.get('apikey')
        if(checkAPIkey(apikey) == True):
            if input_text is None or len(input_text) < 250:
                return jsonify({"error": " 'text' parameter missing or too short"}), 400
            else:
                average_perplexity, threshold_value,  machine_lines, filtered_text = use_perplexity_model(input_text)
                result = {
                        "average_perplexity": average_perplexity,
                        "threshold_value": threshold_value, 
                        "machine_lines": machine_lines,
                        "filtered_text": filtered_text
                }
                return jsonify({"perplexity_values": result}), 200
        else:
            return jsonify({"error": "access denied"}), 500    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/tfidfngram_classification', methods=['POST'])
def tfidfngram_classification_endpoint():
    try:
        data = request.get_json()
        input_text = data.get('input_data')
        apikey = data.get('apikey')
        if(checkAPIkey(apikey) == True):
            if input_text is None or len(input_text) < 250:
                return jsonify({"error": " 'text' parameter missing or too short"}), 400
            else:
                classification = tf_idf_n_gram_model(input_text)
                classification_number = classification[0].tolist()
                classification_text = classification[1]
                classification_probability = classification[2].tolist()
                result = {
                    "classification_number": classification_number,
                    "classification_text": classification_text,
                    "classification_probability":classification_probability
                }
                return jsonify({"tfidf_classification": result}), 200
        else:
            return jsonify({"error": "access denied"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/tfidf_classification', methods=['POST'])
def tfidf_classification_endpoint():
    try:
        data = request.get_json()
        input_text = data.get('input_data')
        apikey = data.get('apikey')
        if(checkAPIkey(apikey) == True):
            if input_text is None or len(input_text) < 250:
                return jsonify({"error": " 'text' parameter missing or too short"}), 400
            else:
                classification = tf_idf_model(input_text)
                classification_number = classification[0].tolist()
                classification_text = classification[1]
                classification_probability = classification[2].tolist()
                result = {
                    "classification_number": classification_number,
                    "classification_text": classification_text,
                    "classification_probability":classification_probability
                }
                return jsonify({"tfidf_classification": result}), 200
        else:
            return jsonify({"error": "access denied"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/burstiness_classification', methods=['POST'])
def burstiness_classification_endpoint():
    try:
        data = request.get_json()
        input_text = data.get('input_data')
        apikey = data.get('apikey')
        if(checkAPIkey(apikey) == True):
            if input_text is None or len(input_text) < 250:
                return jsonify({"error": " 'text' parameter missing or too short"}), 400
            else:
                classification = burstiness_model(input_text)
                classification_number = classification[0].tolist()
                classification_text = classification[1]
                classification_probability = classification[2].tolist()
                result = {
                    "classification_number": classification_number,
                    "classification_text": classification_text,
                    "classification_probability":classification_probability
                }
                return jsonify({"burstiness_classification": result}), 200
        else:
            return jsonify({"error": "access denied"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def checkAPIkey(APIkey):
    #Implement in more detail if time, only basic implementation for now
    if APIkey == "securekey":
        return True
    else:
        return False
    
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5001, debug=True)

