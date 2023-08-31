# AVUS-Submission
Installation instructions

To run the system locally the following software is required:
1.	A modern HTML5 browser. (Project was created using Firefox and tested with Chrome.
2.	Python 3
3.	Visual Studio code
  a.	VS Code Python extension:
  b.	Guide for setting Python up in VS Code: https://code.visualstudio.com/docs/python/python-tutorial
4.	Nvidia Cuda
  a.	Requires a CUDA enabled GPU 
  b.	Nvidia Display drivers compatible with Cuda Toolkit
  c.	https://developer.nvidia.com/cuda-toolkit
5.	Installed all the libraries listed in requirement.txt 
Once the above software is installed, follow the instructions below (Only tested for windows):
1)	Extract the compressed files while maintaining the file structure.
2)	Open Visual Studio Code, click the “File” button on the top bar and go down to “Open Folder…”
3)	File Explorer will be brought up, go to the folder the files were extracted to and click on the top file “AVUS”. Visual Studio Code should open the file. Under the heading “AVUS” there should folders named “classification_model_code”, “Datasets”, “templates”, “Testing”, “static”, “trained_classification_models” and files named “.env”, “API.py”, “app.py”, “burstiness_model.py”, “classification_model.py”, “errorhandling.py”, “perplexity_model.py”, “prompting_gpt.py”, “requirement.txt” and “test_performance.py”. These are the main files and folders.
4)	Use the short cuts ctrl + shift + p, and a search bar should appear at the top of the screen. Type “Python: Select Interpreter”, select python 3.11.4 or later (not tested with earlier versions)
5)	Use the shortcut ctrl + shift + p, and a search bar should appear at the top of the screen. Type “Python: Create Environment” and select Venv, python 3.11.4 or later (not tested with earlier versions) and do -not- select any dependencies (Installed without errors if command given directly to console)
6)	Go to “Terminal” in the top navbar, and select “New Terminal”. Ensure the new terminal is in focus and not any old ones. There should be a line that reads “(.venv) PS file location” with the (.venv) in green
7)	Type into the console “pip install -r requirement.txt” and hit enter.
8)	Once installation is finished type in “pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117”
9)	All of the packages should be installed now.
10)	To run the webapp and detection tool, two files need to be run API.py and app.py at the same time.
11)	To run the API.py right click on it and click on the “Run Python File in Terminal”, the API will load and run
12)	To run the app.py (or API.py if you ran app.py first) is slightly more complicated as click the same buttons as point 11 will lead the file to attempt to run after the API.py is finished. Instead a new terminal needs to be created and the follow command input “(.venv) PS C:\Users\conor\Desktop\Test Install\AVUS> & "c:/Users/conor/Desktop/Test Install/AVUS/.venv/Scripts/python.exe" "c:/Users/conor/Desktop/Test Install/AVUS/app.py"” the command starts after the AVUS> so type the add forward, change the file location to point to where your app.py is installed.
13)	Now that both app.py and API.py are running, the webapp can be accessed on the following URI:
  a.	http://127.0.0.1:5000/
14)	The API is called by the webapp when used, or if required can be accessed through apps such as Postman provided a Json file is included with the end point
  a.	End point example: http://127.0.0.1:5001/all_results
