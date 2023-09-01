import openai
import random
import os
import errorhandling

from dotenv import load_dotenv, find_dotenv

#Loads the dotenv that stores the openai apikey
_ = load_dotenv(find_dotenv()) # read local .env file


#Checks if the user has passed an APIkey, if not it loads the one from the dotenv.
def inputkey(apikey):
    print(apikey)
    if(apikey != ""):
        openai.api_key = apikey
    else:
        openai.api_key  = os.getenv('OPENAI_API_KEY')
        

#Creates a prompt for the request model with a passed temperature, then passes it to the OpenAI API to get a result
def get_response(prompt, model, temperature = 0.7):
    try:
        messages = [{"role": "user", "content": prompt}]
        if model == "gpt-3.5-turbo" or model == "gpt-4":
            response = openai.ChatCompletion.create(
                model = model,
                messages = messages,
                temperature = temperature, # this is the degree of randomness of the model's output
                
            )
            return response.choices[0].message["content"]
        else:
            response = openai.Completion.create(
                engine = model,
                prompt = prompt,
                temperature = temperature,
                max_tokens = 4000
            )
            return response.choices[0].text.strip()
    except openai.OpenAIError as authentication:
        raise errorhandling.APIKeyAuthenticationError("The API key failed to authenticate with OpenaAI.")

    

#This function is used to make a text prompt
def prompt(model):
    size = random.randrange(3, 5)
    topic = random.randrange(0, 15)
    topics = ["Science", "Mathematics", "Physics", "Gaming", "Computer Science", "Short Story", "Engineering", "Hiking", "Map Making", "Cooking", "recipes", "Medieval History", "Roman History", "Greek History", "Viking History"]
    size_string = str(size)
    prompt = f"""
    Your task is to generate a short summary about {size_string} paragraphs long,
    the topic is {topics[topic]},
    bullet point format should not be used. The text should be a minimum of 250 chars
    and a maximum of 5000 chars. No inappropriate topics should be used, nor inappropriate language.
    Do not include "Topic: " at the top.
    """
    response = get_response(prompt, model)
    return response


    
    