import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import html
import time
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')



def getPageComments(pageNumber):
    url = "https://www.walmart.com/seller/17602?page=" + str(pageNumber) # Example product page

    # Headers to mimic a real browser request
    headers = {
        "User-Agent": "Mozilla/6.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # Send a GET request to the URL
    response = requests.get(url, headers=headers)
    #print(response.content)
    #soup = BeautifulSoup(response.content, "html.parser")
    #index = response.content.find('<ul class="cc-2 cg-4 pl0 pr2 mt4">')
    temp =  response.content.decode('UTF-8')
    index = temp.find('<ul class="cc-2 cg-4 pl0 pr2 mt4">')
    temp = temp[index:]

    index = temp.find('<div class="w_DHV_ pv3 mv1">')
    temp = temp[index:]

    result = []

    counter = 0
    while temp.find('<div class="w_DHV_ pv3 mv1">') != -1:
        
        startIndex = temp.find('<div class="w_DHV_ pv3 mv1">')
        endIndex = temp.find('<div class="w_al6g overflow-visible h-100">')
        #print(startIndex, endIndex)

        comment = temp[startIndex:endIndex]
        startIndexForStar = comment.find('<span class="w_iUH7">')
        comment = comment[startIndexForStar + len('<span class="w_iUH7">'):]
        endIndexForStar = comment.find(' out')
        starValue = comment[:endIndexForStar] # varan 1
        #print(starValue)

        dateStartIndex = comment.find('<div class="f7 gray mt1">')
        comment = comment[dateStartIndex + len('<div class="f7 gray mt1">'):]
        dateEndIndex = comment.find('</div>')

        dateValue = comment[:dateEndIndex]
        #print(dateValue) # varan 2

        reviewStartIndex = comment.find('<b></b>')
        #print(reviewStartIndex)
        #reviewValue = ""
        if (reviewStartIndex != -1):
            comment = comment[reviewStartIndex + len('<b></b>'):]
            reviewEndIndex = comment.find('</span>')
            reviewValue = comment[:reviewEndIndex] # varan 3
            reviewValue = html.unescape(reviewValue)
            #print(html.unescape(reviewValue))
        else:
            reviewValue = "No review"
        result.append({"date": dateValue, "star": int(starValue), "review": reviewValue})

        temp = temp[endIndex + len('<div class="w_al6g overflow-visible h-100">'):]
        counter += 1

    #print(result)
    return result

allComments = [];
pageNumber = 1;
while True:
    res = getPageComments(pageNumber)
    #time.sleep(2)
    if (res == []):
        break
    allComments.extend(res)
    pageNumber += 1    
print(allComments)
print(len(allComments))




df = pd.DataFrame(allComments)



def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    
    # Remove punctuation
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Rejoin the tokens into a single string
    return ' '.join(tokens)


df['cleaned_review'] = df['review'].apply(preprocess_text)



# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()



# Define a function to analyze sentiment
def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores


df['sentiment'] = df['cleaned_review'].apply(analyze_sentiment)


def categorize_sentiment(row):
    compound_score = row['sentiment']['compound']  # Access the compound score
    star_rating = row['star']  # Access the star rating

    # Define the sentiment label based on compound score and star rating
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        # Handle neutral sentiment by looking at the star rating
        if star_rating in [4, 5]:
            return 'Positive'
        elif star_rating == 3:
            return 'Neutral'
        else:
            return 'Negative'


# Apply categorization to each row in the DataFrame
df['sentiment_label'] = df.apply(categorize_sentiment, axis=1)

# Inspect the sentiment labels
print(df[['cleaned_review', 'sentiment', 'sentiment_label']].head())










