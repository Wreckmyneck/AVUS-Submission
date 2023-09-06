import os
import openai
import random

openai.api_key = "" #need to add an api key

def save_text_to_file(text, index):
    file_path = r"TestWithoutOverWriting.csv"
    
    # Check if the file exists, if not, create it
    if not os.path.exists(file_path):
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                textcreate = "index,, content\n"
                file.write(textcreate)
        except IOError:
            print("Error: Unable to create file", file_path)
            return

    # Add content and count at the given index
    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            text = str(index) + ",," + text + "\n"
            file.write(text)
        print("Text saved successfully to", file_path)
    except IOError:
        print("Error: Unable to save text to", file_path)

def article_generator(amount):
    # Array of news article topics/categories
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

    dialogue1 = [
        {"role": "system", "content": "You are a journalist that writes for a online newspaper."},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "I will write a news paper article for an online newspaper for you. Anything else needed?"},
        {"role": "user", "content": ""}
    ]

    for j in range(0, amount):
        selected_topic = random.choice(news_topics)
        selected_company = random.choice(news_companies)
        dialogue1[1]["content"] = f"I want you to write me an article on {selected_topic}. You will not use any filler text, you will only print text related to the article. This includes confirmation messages"
        dialogue1[3]["content"] = f"Yes, I do not need a heading or title or anything. You are writing for the {selected_company} so write in their style. Pretend as if you are inserting images by adding image captions."
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=dialogue1,
            temperature=0.6
        )

        generated_text = res.choices[0].message.content
        text = replace_newlines(generated_text)
        print(f'Generating article number {j}')
        save_text_to_file(text, j)

def abstract_generator(amount):
    # Array of news article topics/categories
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

    dialogue1 = [
        {"role": "system", "content": "You are an assistant for a scientific researcher. Your job is to create abstracts"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "I will create an abstract for you now. Are there any other conditions?"},
        {"role": "user", "content": "Yes, I don't want a title or any confirmation, just the abstract."}
    ]

    for j in range(0, amount):
        selected_topic = random.choice(research_paper_topics)
        dialogue1[1]["content"] = f"I have just written a scientific paper on {selected_topic}, and I need an abstract to be made on this topic."
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=dialogue1,
            temperature=0.6
        )

        generated_text = res.choices[0].message.content
        text = replace_newlines(generated_text)
        print(f'Generating article number {j}')
        save_text_to_file(text, j)

def review_generator(amount):
    # Array of news article topics/categories
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

    dialogue1 = [
        {"role": "system", "content": "You are a person who just recently stayed at a hotel and desire to write a review"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "The review I am writing will be no more than 500 words"},
        {"role": "user", "content": "The review must not include dates."}
    ]

    for j in range(0, amount):
        selected_hotel = random.choice(hotel_names)
        selected_sentiment = random.choice(sentiments)
        dialogue1[1]["content"] = f"You recently stayed at a hotel called {selected_hotel}. You are {selected_sentiment} about your stay at the hotel"
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=dialogue1,
            temperature=0.6
        )

        generated_text = res.choices[0].message.content
        text = replace_newlines(generated_text)
        print(f'Generating article number {j}')
        save_text_to_file(text, j)

def blogging_generator(amount):
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

    dialogue1 = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Your blogs consist of around 500-1500 words depending on the topic."},
        {"role": "assistant", "content": "I will write my blog post now."},
        {"role": "user", "content": ""}
    ]

    for j in range(0, amount):
        selected_blogging_topic = random.choice(blog_topics)
        selected_website = random.choice(blogging_websites)
        selected_mood = random.choice(moods)
        dialogue1[0]["content"] = f"You are a person who writes blogs about {selected_blogging_topic} on the website {selected_website}."
        dialogue1[3]["content"] = f"You are currently feeling {selected_mood} and will write with that in mind."
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=dialogue1,
            temperature=0.6
        )

        generated_text = res.choices[0].message.content
        text = replace_newlines(generated_text)
        print(f'Generating article number {j}')
        save_text_to_file(text, j)


def replace_newlines(text):
    return text.replace('\n', '%%newline%%')

article_generator(1)
abstract_generator(1)
review_generator(1)
blogging_generator(1)





