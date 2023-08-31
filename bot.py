import csv
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load training data from the CSV file
def load_training_data(filename):
    training_data = []
    with open(filename, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            intent, question, response = row
            training_data.append((intent, question, response))
    return training_data

# Function to preprocess text using NLTK
def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text.casefold()

# Function to get the best matching response based on intent and question
def get_response(intent, processed_question, training_data):
    best_match_score = 0
    best_response = "Sorry, I don't understand your question."

    for data_intent, data_question, response in training_data:
        #print("Intent: ", intent, ", Data_intent: ", data_intent, ", Data_question: ", data_question, ", response: ", response)
        #print("intent: ", intent, ", data_intent: ", data_intent, ", result: ", (intent==data_intent))
        if intent == data_intent:
            # Preprocess data question
            preprocessed_data_question = preprocess_text(data_question)

            # Calculate the match score based on intent and preprocessed question similarity
            match_score = calculate_match_score(processed_question, preprocessed_data_question)
            
            # Update the best response if the match score is higher
            if match_score > best_match_score:
                best_match_score = match_score
                best_response = response

    return best_response

# Function to calculate the similarity match score between user question and data question
def calculate_match_score(user_question, data_question):
    user_words = set(user_question.lower().split())
    data_words = set(data_question.lower().split())
    
    common_words = user_words.intersection(data_words)
    match_score = len(common_words)
    return match_score


# Function to get the intent based on user input
def get_intent(user_input, rules):
    for intent, rule in rules.iterrows():
        match = re.match(rule["Rule"], user_input)
        if match:
            return rule["Intent"]
    return None

# Load intent rules from a CSV file
rules = pd.read_csv('intent_rule.csv')
print(rules)

# Load training data
training_data = load_training_data('AppointmentTraining.csv')

# ... (your imports and functions)

print("Chatbot: Hi there! How can I assist you today?")
while True:
    user_input = input("You: ")

    # Preprocess user input
    preprocessed_user_input = preprocess_text(user_input)
    
    # Extract intent and user question
    intent = get_intent(preprocessed_user_input, rules)
    user_question = preprocessed_user_input

    # Get the best response based on intent and question
    best_response = get_response(intent, user_question, training_data)
    
    print("Chatbot:", best_response)

    if intent == "exit":
        print("Chatbot: Goodbye! Have a great day!")
        break
