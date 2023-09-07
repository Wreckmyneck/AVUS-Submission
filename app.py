import random
import string
import PyPDF2
from docx import Document
from flask import Flask, request, render_template, flash, redirect, url_for, session
import requests
from werkzeug.utils import secure_filename
from flask_session import Session
import prompting_gpt
import errorhandling


# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'thisisverysecret'
# Initialize the GPT prompter
prompter = prompting_gpt
# Set of allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
# Configure session settings
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
# Route to the home page
@app.route('/')
def index():
    # Generate a session ID if not present
    if not session.get("session_id"):
        id = generate_random_string_for_session(16)
        session["session_id"] = id

    # Check if returning from another page
    returning_from_other_page = request.args.get('return')
    if returning_from_other_page == "1":
        # Display the result using the session data
        average_perplexity, final_result, model2_text, model3_text, burstiness_text, machine_lines, filtered_text, burstiness_prob_ai, burstiness_prob_human, model2_prob_ai, model2_prob_human, model3_prob_ai, model3_prob_human = display_result()
        return render_template('index.html',  
                               average_perplexity=average_perplexity,
                               final_result=final_result,
                               model2_classification=model2_text,
                               model3_classification=model3_text,
                               machine_lines_as_string=machine_lines,
                               burstiness_text = burstiness_text,
                               filtered_text=filtered_text, 
                               burstiness_prob_AI = burstiness_prob_ai,
                               burstiness_prob_human =burstiness_prob_human, 
                               model2_prob_AI = model2_prob_ai, 
                               model2_prob_human = model2_prob_human, 
                               model3_prob_AI =model3_prob_ai, 
                               model3_prob_human= model3_prob_human)
    else:
        return render_template('index.html')

# Route to the "How To" page
@app.route('/how-to')
def how_to():
    # Render the "how-to.html" template to display instructions
    return render_template("how-to.html")

# Route to explain perplexity
@app.route('/perplexity_explanation')
def perplexity_explanation():
    # Retrieve necessary data from the session
    average_perplexity, final_result, model2_text, model3_text, burstiness_text, machine_lines, filtered_text, burstiness_prob_ai, burstiness_prob_human, model2_prob_ai, model2_prob_human, model3_prob_ai, model3_prob_human = display_result()

    # Render the "perplexity_explanation.html" template with average_perplexity data
    return render_template('perplexity_explanation.html', average_perplexity=average_perplexity)

# Route to explain TF-IDF N-gram classification
@app.route('/tfidf_ngram_classification_explanation')
def tfidf_ngram():
    # Retrieve necessary data from the session
    average_perplexity, final_result, model2_text, model3_text, burstiness_text, machine_lines, filtered_text, burstiness_prob_ai, burstiness_prob_human, model2_prob_ai, model2_prob_human, model3_prob_ai, model3_prob_human = display_result()

    # Render the "tfidf_ngram_classification_explanation.html" template with model2_text data
    return render_template('tfidf_ngram_classification_explanation.html', model2_text=model2_text, model2_prob_ai = model2_prob_ai, model2_prob_human = model2_prob_human)

# Route to explain TF-IDF classification
@app.route('/tfidf_classification_explanation')
def tfidf():
    # Retrieve necessary data from the session
    average_perplexity, final_result, model2_text, model3_text, burstiness_text, machine_lines, filtered_text, burstiness_prob_ai, burstiness_prob_human, model2_prob_ai, model2_prob_human, model3_prob_ai, model3_prob_human = display_result()

    # Render the "tfidf_classification_explanation.html" template with model3_text data
    return render_template('tfidf_classification_explanation.html', model3_text=model3_text, model3_prob_ai = model3_prob_ai, model3_prob_human = model3_prob_human)

#Route to explain Burstiness classification
@app.route('/burstiness_classification_explanation')
def burstiness():
    # Retrieve necessary data from the session
    average_perplexity, final_result, model2_text, model3_text, burstiness_text, machine_lines, filtered_text, burstiness_prob_ai, burstiness_prob_human, model2_prob_ai, model2_prob_human, model3_prob_ai, model3_prob_human = display_result()

    # Render the "tfidf_classification_explanation.html" template with model3_text data
    return render_template('burstiness_classification_explanation.html', burstiness_text=burstiness_text, burstiness_prob_ai = burstiness_prob_ai, burstiness_prob_human = burstiness_prob_human)

#Route to about page
@app.route('/about')
def about():
   return render_template('about.html')

# Route for the main functionality
@app.route('/', methods=['POST', 'GET'])
def button_functions():
    response = ""
    promptmodel = ""

    # Handle POST request
    if request.method == 'POST':
        apikey = request.form.get('apikeyinput') # for openai if input by user
        try:
            action = request.form.get('action')

            # Generate text using different GPT models
            if action == 'Generate ChatGPT(3.5) Text':
                print("Response may take some time")
                prompter.inputkey(apikey)
                promptmodel = "gpt-3.5-turbo"
                response = prompter.prompt(promptmodel)
                return render_template('index.html', response = response)
            elif action == 'Generate GPT2 Text':
                print("Response may take some time")
                prompter.inputkey(apikey)
                promptmodel = "text-davinci-002"
                response = prompter.prompt(promptmodel)
                return render_template('index.html', response = response)
            elif action == 'Generate GPT3 Text':
                print("Response may take some time")
                prompter.inputkey(apikey)
                promptmodel = "text-davinci-003"
                response = prompter.prompt(promptmodel)
                return render_template('index.html', response = response)
            elif action == 'Generate GPT4 Text':
                print("Response may take some time")
                prompter.inputkey(apikey)
                promptmodel = "gpt-4"
                response = prompter.prompt(promptmodel)
                return render_template('index.html', response = response)
            # Process input text or file
            elif action == 'Process Text':
               text = request.form.get('textboxinput')
               file = request.files['file']

               # Handle file input
               if file.filename and text == '':
                  try:
                     if file and allowed_file(file.filename):
                           filename = secure_filename(file.filename)
                           if file.filename.endswith('.pdf'):
                              # Extract text from PDF, and pass it to the function that calls teh api to get results, then display results.
                              pdf = PyPDF2.PdfReader(file)
                              text = ''
                              for page in range(len(pdf.pages)):
                                 text += pdf.pages[page].extract_text()
                              text = ''.join(filter(lambda x: x.isprintable(), text))
                              character_count = len(text)
                              if character_count < 250:
                                 flash('The text is too short. It must be a minimum of 250 characters', 'inputerror')
                                 return redirect(url_for('index'))
                              else:
                                 api_url = "http://127.0.0.1:5001/all_results"
                                 data = {
                                    "input_data":text,
                                    "apikey":"securekey"
                                 }
                                 getresults(api_url, data)
                                 average_perplexity, final_result, model2_text, model3_text, burstiness_text, machine_lines, filtered_text, burstiness_prob_ai, burstiness_prob_human, model2_prob_ai, model2_prob_human, model3_prob_ai, model3_prob_human = display_result()
                                 return render_template("index.html", average_perplexity=average_perplexity, final_result=final_result, model2_classification=model2_text, model2_prob_ai = model2_prob_ai, model2_prob_human = model2_prob_human, model3_classification=model3_text, model3_prob_ai = model3_prob_ai, model3_prob_human = model3_prob_human,burstiness_text = burstiness_text, burstiness_prob_ai = burstiness_prob_ai, burstiness_prob_human = burstiness_prob_human, machine_lines=machine_lines, filtered_text=filtered_text, file = True) 
                           elif file.filename.endswith('.docx'):
                              # Extract text from Docx, and pass it to the function that calls teh api to get results, then display results.
                              doc = Document(file)
                              text = ""
                              for paragraph in doc.paragraphs:
                                 text += paragraph.text
                              character_count = len(text)
                              if character_count < 250:
                                 flash('The text is too short. It must be a minimum of 250 characters', 'inputerror')
                                 return redirect(url_for('index'))
                              else:
                                 api_url = "http://127.0.0.1:5001/all_results"
                                 data = {
                                    "input_data":text,
                                    "apikey":"securekey"
                                 }
                                 getresults(api_url, data)
                                 average_perplexity, final_result, model2_text, model3_text, burstiness_text, machine_lines, filtered_text, burstiness_prob_ai, burstiness_prob_human, model2_prob_ai, model2_prob_human, model3_prob_ai, model3_prob_human = display_result()
                                 return render_template("index.html", average_perplexity=average_perplexity, final_result=final_result, model2_classification=model2_text, model2_prob_ai = model2_prob_ai, model2_prob_human = model2_prob_human, model3_classification=model3_text, model3_prob_ai = model3_prob_ai, model3_prob_human = model3_prob_human,burstiness_text = burstiness_text, burstiness_prob_ai = burstiness_prob_ai, burstiness_prob_human = burstiness_prob_human, machine_lines=machine_lines, filtered_text=filtered_text, file = True)  
                           elif file.filename.endswith('.txt'):
                              text = file.read().decode('utf-8')
                              character_count = len(text)
                              if character_count < 250:
                                 flash('The text is too short. It must be a minimum of 250 characters', 'inputerror')
                                 return redirect(url_for('index'))
                              else:
                                 # Process text using models and display results through API, and pass it to the function that calls teh api to get results, then display results.
                                 api_url = "http://127.0.0.1:5001/all_results"
                                 data = {
                                    "input_data":text,
                                    "apikey":"securekey"
                                 }
                                 getresults(api_url, data)
                                 average_perplexity, final_result, model2_text, model3_text, burstiness_text, machine_lines, filtered_text, burstiness_prob_ai, burstiness_prob_human, model2_prob_ai, model2_prob_human, model3_prob_ai, model3_prob_human = display_result()
                                 return render_template("index.html", average_perplexity=average_perplexity, final_result=final_result, model2_classification=model2_text, model2_prob_ai = model2_prob_ai, model2_prob_human = model2_prob_human, model3_classification=model3_text, model3_prob_ai = model3_prob_ai, model3_prob_human = model3_prob_human,burstiness_text = burstiness_text, burstiness_prob_ai = burstiness_prob_ai, burstiness_prob_human = burstiness_prob_human, machine_lines=machine_lines, filtered_text=filtered_text, file = True)  
                     else:
                        flash('Please input a docx, pdf or txt file only', 'inputerror')
                        return redirect(url_for('index'))
                  except KeyError:
                     pass
                  except Exception as e:
                     flash(f'An error occurred while processing the file input', 'error')
                     print(f"An error occurred: {str(e)}")
                     return redirect(url_for('index'))
                  
                  # Handle text input, and pass it to the function that calls teh api to get results, then display results.
               elif text != '' and file.filename == '':
                  try:
                     character_count = len(text)
                     if 250 <= character_count <= 5000:
                        api_url = "http://127.0.0.1:5001/all_results"
                        data = {
                           "input_data":text,
                           "apikey":"securekey"
                        }
                        getresults(api_url, data)
                        average_perplexity, final_result, model2_text, model3_text, burstiness_text, machine_lines, filtered_text, burstiness_prob_ai, burstiness_prob_human, model2_prob_ai, model2_prob_human, model3_prob_ai, model3_prob_human = display_result()
                        return render_template("index.html", average_perplexity=average_perplexity, final_result=final_result, model2_classification=model2_text, model2_prob_ai = model2_prob_ai, model2_prob_human = model2_prob_human, model3_classification=model3_text, model3_prob_ai = model3_prob_ai, model3_prob_human = model3_prob_human,burstiness_text = burstiness_text, burstiness_prob_ai = burstiness_prob_ai, burstiness_prob_human = burstiness_prob_human, machine_lines=machine_lines, filtered_text=filtered_text, file=False)
 
                     elif character_count < 250:
                        flash('The text is too short. It must be a minimum of 250 characters', 'inputerror')
                        return redirect(url_for('index'))
                           
                     else:
                        flash('The text is too long. It must be a maximum of 5000 characters. If you need more, please input a file.', 'inputerror')
                        return redirect(url_for('index'))
                     
                     # Handle exceptions
                  except Exception as e:
                     flash(f'An error occurred while processing the textbox input', 'error')
                     print(f"An error occurred: {str(e)}")
                     return redirect(url_for('index'))
                  
                  # Handle cases when both text and file are provided
               else:
                  flash('Please input either text in the textbox or a file. Not both.', 'inputerror')
                  return redirect(url_for('index'))
            
            # Handle other actions or display default template
            else:
               return render_template("index.html")

        # Handle API key authentication errors
        except errorhandling.APIKeyAuthenticationError as authentication:
            flash('Authentication error with openAI API key. Please input your own. If you are using gpt-4, ensure your account is openai premium.', 'error')
            return redirect(url_for('index'))
        
        # Handle other exceptions
        except Exception as e:
            flash(f'An error occurred while processing the request', 'error')
            print(f"An error occurred: {str(e)}")
            return redirect(url_for('index'))

    # Display the default template for GET requests
    return render_template("index.html")

#This is used to generate a string for the session
def generate_random_string_for_session(length):
   characters = string.ascii_letters + string.digits
   return ''.join(random.choice(characters) for _ in range(length))

#This function is used to send json files to an end point, and get all the returned results from the endpoint, then store them as part of the session so they can be accessed on any page
def getresults(endpoint, data):
   response = requests.post(endpoint, json=data)
   if response.status_code == 200:
      result_json = response.json()
      result = result_json["full_results"]
      print(result)
      average_perplexity = result.get("average_perplexity")
      threshold_value = result.get("threshold_value")
      model2_binary_number = result.get("model2_binary_number")
      model2_text = result.get("model2_text")
      model2_probability = result.get("model2_probability")
      model2_prob_AI = model2_probability[0][0]
      model2_prob_human = model2_probability[0][1]
      model3_binary_number = result.get("model3_binary_number")
      model3_text = result.get("model3_text")
      model3_probability = result.get("model3_probability")
      model3_prob_AI = model3_probability[0][0]
      model3_prob_human = model3_probability[0][1]
      burstiness_binary_number = result.get("burstiness_number")
      burstiness_probability = result.get("burstiness_probability")
      burstiness_prob_AI = burstiness_probability[0][0]
      burstiness_prob_human = burstiness_probability[0][1]
      burstiness_text = result.get("burstiness_text")
      machine_lines = result.get("machine_lines")
      filtered_text = result.get("filtered_text")
      final_result = result.get("main_result")
      store_results(average_perplexity, final_result, model2_text, model3_text, burstiness_text, machine_lines, filtered_text, burstiness_prob_AI, burstiness_prob_human, model2_prob_AI, model2_prob_human, model3_prob_AI, model3_prob_human)
   else:
      flash('API error', 'inputerror')
      return redirect(url_for('index')) 

#This is used to determine if a file type is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#This function is used to store the values as part of the session under the same name
def store_results(average_perplexity, final_result, model2_text, model3_text, burstiness_text, machine_lines, filtered_text, burstiness_prob_ai, burstiness_prob_human, model2_prob_ai, model2_prob_human, model3_prob_ai, model3_prob_human):
   # Store the variables in the session
    session['average_perplexity'] = average_perplexity
    session['final_result'] = final_result
    session['model2_text'] = model2_text
    session['model3_text'] = model3_text
    session['machine_lines'] = machine_lines
    session['filtered_text'] = filtered_text
    session['burstiness_text'] = burstiness_text
    session['burstiness_prob_ai'] = burstiness_prob_ai
    session['burstiness_prob_human'] = burstiness_prob_human
    session['model2_prob_ai'] = model2_prob_ai
    session['model2_prob_human'] = model2_prob_human
    session['model3_prob_ai'] = model3_prob_ai
    session['model3_prob_human'] = model3_prob_human

#This function is used to find the variables stored as part of the session, and return them when called.
def display_result():
    # Retrieve the variables from the session
    average_perplexity = session.get('average_perplexity')
    final_result = session.get('final_result')
    model2_text = session.get('model2_text')
    model3_text = session.get('model3_text')
    burstiness_text = session.get('burstiness_text')
    machine_lines = session.get('machine_lines')
    filtered_text = session.get('filtered_text')
    burstiness_prob_ai = session.get('burstiness_prob_ai')
    burstiness_prob_human = session.get('burstiness_prob_human')
    model2_prob_ai = session.get('model2_prob_ai')
    model2_prob_human = session.get('model2_prob_human')
    model3_prob_ai = session.get('model3_prob_ai')
    model3_prob_human = session.get('model3_prob_human')
    return average_perplexity, final_result, model2_text, model3_text, burstiness_text, machine_lines, filtered_text, burstiness_prob_ai, burstiness_prob_human, model2_prob_ai, model2_prob_human, model3_prob_ai, model3_prob_human 

if __name__ == "__main__":
   app.run(debug = True)
