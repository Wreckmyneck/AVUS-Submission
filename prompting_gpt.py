import openai
import random
import os
import errorhandling

from dotenv import load_dotenv, find_dotenv

#Loads the dotenv that stores the openai apikey
_ = load_dotenv(find_dotenv()) # read local .env file


#Checks if the user has passed an APIkey, if not it loads the one from the dotenv.
def inputkey(apikey):
    if(apikey != ""):
        openai.api_key = apikey
    else:
        openai.api_key  = os.getenv('OPENAI_API_KEY')
        

#Creates a prompt for the request model with a passed temperature, then passes it to the OpenAI API to get a result
def get_response(prompt, model, temperature = 0.7):
    try:
        
        if model == "gpt-3.5-turbo" or model == "gpt-4":
            response = openai.ChatCompletion.create(
                model = model,
                messages = prompt,
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
    prompt = make_prompt(model)
    response = get_response(prompt, model)
    return response

def make_prompt(model):
    try:
        type_of_text = [
            "blog post",
            "news paper",
            "abstract",
            "review",
        ]
        type_prompt = random.choice(type_of_text)
        if type_prompt == "blog post":
            blogging_websites = [
            "WordPress",
            "Reddit",
            "Twitter",
            "Blogger",
            "Medium",
            "Tumblr",
            "Wix",
            "Squarespace",
            "Ghost",
            "Weebly",
            "HubSpot",
            "Joomla",
            "Typepad",
            "Drupal",
            "LiveJournal",
            "Strikingly",
            "Postach.io",
            "Write.as",
            "Substack",
            "Jekyll",
            "Webflow",
            "Hashnode"
            ]

            blog_topics = [
                "Travel",
                "Technology",
                "Health and Wellness",
                "Food and Cooking",
                "Personal Development",
                "Fashion and Style",
                "Finance and Investing",
                "Home Decor and DIY",
                "Parenting and Family",
                "Fitness and Exercise",
                "Education and Learning",
                "Entertainment and Pop Culture",
                "Art and Creativity",
                "Sports and Fitness",
                "Science and Innovation",
                "Business and Entrepreneurship",
                "Social Media and Marketing",
                "Books and Literature",
                "Environment and Sustainability",
                "Current Events and News"
            ]

            moods = [
                "Happy", 
                "Sad", 
                "Excited", 
                "Angry", 
                "Relaxed", 
                "Anxious", 
                "Confused", 
                "Content", 
                "Bored", 
                "Energetic", 
                "Peaceful", 
                "Grateful"
                ]
            selected_blogging_topic = random.choice(blog_topics)
            selected_website = random.choice(blogging_websites)
            selected_mood = random.choice(moods)
            if model == "gpt-3.5-turbo" or model == "gpt-4":

                prompt = [
                    {"role": "system", "content": f"You are a person who writes blogs about {selected_blogging_topic} on the website {selected_website}."},
                    {"role": "user", "content": "Your blogs consist of around 500-1500 words depending on the topic but never more than 4500 characters."},
                    {"role": "assistant", "content": "I will write my blog post now."},
                    {"role": "user", "content": f"You are currently feeling {selected_mood} and will write with that in mind."}
                ]
                return prompt
            else:
                prompt = f"You are a person who writes blogs about {selected_blogging_topic} on the website {selected_website}. You are currently feeling {selected_mood} and will write with that in mind. You will not write more than 4500 characters."
                return prompt
        elif type_prompt == "news paper":
            news_topics = [
                    "Politics",
                    "Business",
                    "Technology",
                    "Health",
                    "Science",
                    "Entertainment",
                    "Sports",
                    "Environment",
                    "World News",
                    "Education"
                ]
            news_companies = [
                "The New York Times",
                "The Washington Post",
                "CNN",
                "The Guardian",
                "MSNBC",
                "Los Angeles Times",
                "BBC News",
                "Reuters",
                "The Huffington Post",
                "NPR",
                "euronews",
                "The Wall Street Journal",
                "Fox News",
                "The National Review",
                "The American Conservative",
                "The Daily Caller",
                "The Economist",
                "The American Spectator",
                "The Daily Telegraph",
                "The Washington Times",
                "The Federalist"
            ]
            selected_topic = random.choice(news_topics)
            selected_company = random.choice(news_companies)
            if model == "gpt-3.5-turbo" or model == "gpt-4":
                prompt = [
                    {"role": "system", "content": "You are a journalist that writes for a online newspaper but never more than 4500 characters."},
                    {"role": "user", "content":  f"I want you to write me an article on {selected_topic}. You will not use any filler text, you will only print text related to the article. This includes confirmation messages"},
                    {"role": "assistant", "content": "I will write a news paper article for an online newspaper for you. Anything else needed?"},
                    {"role": "user", "content": f"Yes, I do not need a heading or title or anything. You are writing for the {selected_company} so write in their style. Pretend as if you are inserting images by adding image captions."}
                ]
                return prompt
            else:
                prompt = f"""You are a journalist that writes for a online newspaper but never more than 4500 characters. I want you to write me an article on {selected_topic}. You will not use any filler text, you will only print text related to the article. This includes confirmation messages. Write in the style of {selected_company} """
                return prompt

        elif type_prompt == "abstract":
            research_paper_topics = [
                "Climate Change and Its Impact on Ecosystems",
                "Artificial Intelligence in Healthcare",
                "The Effects of Social Media on Mental Health",
                "Cybersecurity Challenges in the Modern World",
                "The Role of Women in STEM Fields",
                "Ethical Considerations in Genetic Engineering",
                "The Impact of Automation on Employment",
                "Mental Health Stigma and Its Effects on Treatment",
                "The History and Future of Space Exploration",
                "Environmental Conservation and Sustainable Practices",
                "The Economics of Renewable Energy Sources",
                "The Influence of Literature on Society",
                "Globalization and Its Impact on Culture",
                "The Role of Education in Reducing Income Inequality",
                "The Ethics of Artificial Intelligence and Robotics",
                "Agricultural Sustainability and Food Security",
                "The Psychology of Decision-Making",
                "The History and Impact of Vaccination Programs",
                "The Future of Autonomous Vehicles",
                "Criminal Justice Reform and Racial Disparities"
            ]

            selected_topic = random.choice(research_paper_topics)
            if model == "gpt-3.5-turbo" or model == "gpt-4":        
                prompt = [
                    {"role": "system", "content": "You are an assistant for a scientific researcher. Your job is to create abstracts"},
                    {"role": "user", "content": f"I have just written a scientific paper on {selected_topic}, and I need an abstract to be made on this topic."},
                    {"role": "assistant", "content": "I will create an abstract for you now. Are there any other conditions?"},
                    {"role": "user", "content": "Yes, I don't want a title or any confirmation, just the abstract. Do not write more than 4500 characters"}
                ]
                return prompt
            else:
                prompt = f"I have just written a scientific paper on {selected_topic}, and I need an abstract to be made on this topic. No title or confirmation required. Do not write more than 4500 characters"
                return prompt
        
        elif type_prompt == "review":
            hotel_names = [
            "Luxury Grand Hotel",
            "Cozy Inn Suites",
            "Ocean View Resort",
            "City Center Plaza Hotel",
            "Mountain Retreat Lodge",
            "Sunset Beach Resort",
            "Riverside Manor Hotel",
            "Harbor View Suites",
            "Green Valley Lodge",
            "Urban Oasis Hotel",
            "Palm Paradise Resort",
            "Lakefront Lodge",
            "Historic Downtown Hotel",
            "Tranquil Haven Inn",
            "Skyline Tower Hotel",
            "Desert Mirage Resort",
            "Seaside Villa Hotel",
            "Wilderness Retreat Lodge",
            "Elegant Manor Suites",
            "Tropical Island Resort"
            ]

            sentiments = [
                "Excellent",
                "Good",
                "Average",
                "Poor",
                "Fantastic",
                "Noisy",
                "Terrible",
                "Decent",
                "Relaxing",
                "Disappointing",
                "Convenient",
                "Mixed",
                "Cozy",
                "Delicious",
                "Surprising",
                "Recommended",
                "Subpar",
                "Rejuvenating",
                "Wonderful",
                "Overpriced"
            ]
            selected_hotel = random.choice(hotel_names)
            selected_sentiment = random.choice(sentiments)
            if model == "gpt-3.5-turbo" or model == "gpt-4":  
                prompt = [
                    {"role": "system", "content": "You are a person who just recently stayed at a hotel and desire to write a review no more than 4500 characters"},
                    {"role": "user", "content": f"You recently stayed at a hotel called {selected_hotel}. You are {selected_sentiment} about your stay at the hotel"},
                    {"role": "assistant", "content": "The review I am writing will be no more than 500 words"},
                    {"role": "user", "content": "The review must not include dates."}
                ]
                return prompt
            else:
                prompt = f"You recently stayed at a hotel called {selected_hotel}. You are {selected_sentiment} about your stay at the hotel. You won't write more than 4500 characters."
                return prompt
        else:
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
            print("if statement failed so a generic was given")
            return prompt
    except Exception as e:
        print("An error occured generating a prompt")



    
    