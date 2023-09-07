# AVUS-Submission

## AVUS(AI-Generated Vs Human-Written Text Detection Software)
AVUS contains a webapp for user interaction and an API for integration between the webapp and text detection models. There are currently four different models implemented which are Term-Frequency-Inverse-Document-Frequency with uni-grams and bi-grams, Term-Frequency-Inverse-Document-Frequency with raw Text input, and Burstiness model (sentence lengths made into a frequency distrubution and flattened into features). The current models are trained on abstracts, news articles, Blogger posts, short stories, and reviews (tripadvisor hotels) with GPT generated text and human sources.

## Motivation
AI-generation of Text has gotten more and more popular recently with tools such as ChatGPT being released with tools such as these it is important that there is an ability to tell the difference between human-written text and AI-generated text. Tools have become available such as GPTZero, ZeroGPT and more, except they don't reveal the methods behind their detection software. It was also motivated as a project for the completion of MSc. in Software Systems Development at Queen's University Belfast, alongside a personal interest in the topic.

## Install Guide
### Software Requirements 
To run the system locally the following software is required:
markup: - A modern HTML5 browser. (Project was created using Firefox and tested with Chrome.
-Python 3
- Visual Studio code
  - VS Code Python extension:
  - Guide for setting Python up in VS Code: https://code.visualstudio.com/docs/python/python-tutorial
- Nvidia Cuda
  - Requires a CUDA enabled GPU 
  - Nvidia Display drivers compatible with Cuda Toolkit
  - https://developer.nvidia.com/cuda-toolkit
- Installed all the libraries listed in requirement.txt 

### Installation instructions
Once the above software is installed, follow the instructions below (Only tested for windows):
Markup: - Extract the compressed files while maintaining the file structure.
- Open Visual Studio Code, click the “File” button on the top bar and go down to “Open Folder…”
- File Explorer will be brought up, go to the folder the files were extracted to and click on the top file “AVUS”. Visual Studio Code should open the file. Under the heading “AVUS” there should folders named “classification_model_code”, “Datasets”, “templates”, “Testing”, “static”, “trained_classification_models” and files named “.env”, “API.py”, “app.py”, “burstiness_model.py”, “classification_model.py”, “errorhandling.py”, “perplexity_model.py”, “prompting_gpt.py”, “requirement.txt” and “test_performance.py”. These are the main files and folders.
- Use the short cuts ctrl + shift + p, and a search bar should appear at the top of the screen. Type “Python: Select Interpreter”, select python 3.11.4 or later (not tested with earlier versions)
- Use the shortcut ctrl + shift + p, and a search bar should appear at the top of the screen. Type “Python: Create Environment” and select Venv, python 3.11.4 or later (not tested with earlier versions) and do -not- select any dependencies (Installed without errors if command given directly to console)
- Go to “Terminal” in the top navbar, and select “New Terminal”. Ensure the new terminal is in focus and not any old ones. There should be a line that reads “(.venv) PS file location” with the (.venv) in green
- Type into the console “pip install -r requirement.txt” and hit enter.
- Once installation is finished type in “pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117”
- All of the packages should be installed now.
- To run the webapp and detection tool, two files need to be run API.py and app.py at the same time.
- To run the API.py right click on it and click on the “Run Python File in Terminal”, the API will load and run
- To run the app.py (or API.py if you ran app.py first) is slightly more complicated as click the same buttons as point 11 will lead the file to attempt to run after the API.py is finished. Instead a new terminal needs to be created and the follow command input “(.venv) PS C:\Users\conor\Desktop\Test Install\AVUS> & "c:/Users/conor/Desktop/Test Install/AVUS/.venv/Scripts/python.exe" "c:/Users/conor/Desktop/Test Install/AVUS/app.py"” the command starts after the AVUS> so type the add forward, change the file location to point to where your app.py is installed.
- Now that both app.py and API.py are running, the webapp can be accessed on the following URI:
  - a.	http://127.0.0.1:5000/
- The API is called by the webapp when used, or if required can be accessed through apps such as Postman provided a Json file is included with the end point
  - a.	End point example: http://127.0.0.1:5001/all_results
 
## Overview/Breakdown
Markup: - The folder classification model code, contains all the code that was used to train the three classification models.
- Dataset contains the two datasets used to train the three classification models, as well as two datasets used to validate the entire system.
- The static folder contains the styles.css and templates contain all the .html files that implement JavaScript and Jinja2 code. The templates all use the base.html and use block insertions.
- Testing contains two files that were used to test the system alongside the excel file that stored the results
- The folder "Failed" contains attempted routes to improve upon the classification but the experiments failed. The code is lacking improvement/refinement and lacking comments due to time-frame.
- The folder "trained_classification_models" contains all the binary classification models that are currently implemented into the system.
- app.py contains all the code that is used to run the webapp side of the project. It has a series of routes that handle serving different webpages based on the url. The post handles the generate text and process button. The generate text buttons sends a call to the prompt_gpt.py that handles everything from determing what APIkey to use, to making a prompt, and sending it back to be displayed. The process text figures out whether it is a file or text input, if it is a file it figures which type it is, converts it into a string called text. The text string from the file reading or textbox input are then put in a Json file with the secure key, sent to the function that handles the API calls, results are returned which are stroed in sessions and another function is used to get the results from the session and display to user. Sessions are used to store results so users can navigate to different pages and return to their results as needed.
- API.py contains all the endpoint routes for the API, as well as multiple functions that handle the different types of models that the text needs to be run through.
- There are currently four models used in the classification Perplexity which utilizes HuggingFaces model, Term-Frequency-Inverse-Document-Frequency of Unigrams and bi-grams, Term-Frequency-Inverse-Document-Frequency of raw text, and burstiness based on sentence length distrubutions in the text.

## Potential Improvements
- Expansion of the training dataset to include more topic areas rather than the ones listed above.
- Expansion of the training dataset to include other available AI tools such as Google Bard
- Research into more methods of detecting AI-generated text. Some examples:
  - Fine-tuning a GPT or BERT model at classifying text (Attempts were made but failed to pan out)
  - Exploration into more features that could be used to detect the text such as lingustics, syntax or grammar
  - Exploration into using Cosine Similiarity
  - Sentiment analysis
  - Unsupervised anomoly detection that can be used to detect any deviations from the norm.
 
## Demo Video
https://youtu.be/QjTTssn-bS4


