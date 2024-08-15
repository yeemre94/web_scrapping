import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import html
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to get comments from a URL
def get_page_comments(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    
    comments = []
    # Modify this to target the correct elements in the Walmart review section
    for review in soup.find_all('div', class_='w_DHV_ pv3 mv1'):
        date = review.find('div', class_='f7 gray mt1').text if review.find('div', class_='f7 gray mt1') else "No date"
        star = int(review.find('span', class_='w_iUH7').text.split()[0]) if review.find('span', class_='w_iUH7') else 0
        text = review.find('b').text if review.find('b') else "No review"
        comments.append({"date": date, "star": star, "review": html.unescape(text)})
    
    return comments

# Text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in set(stopwords.words('english'))]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Sentiment analysis function
def analyze_sentiment(text):
    return sia.polarity_scores(text)

# Streamlit UI
st.title("URL-based Sentiment Analysis")

# Input for URL with default value
url_input = st.text_input("Enter the URL for sentiment analysis", "https://www.walmart.com/seller/17602?page=1")

if st.button("Analyze"):
    if url_input:
        # Scrape comments from the input URL
        comments_data = get_page_comments(url_input)
        
        if comments_data:
            df = pd.DataFrame(comments_data)
            df['cleaned_review'] = df['review'].apply(preprocess_text)
            df['sentiment'] = df['cleaned_review'].apply(analyze_sentiment)

            # Categorize sentiment
            def categorize_sentiment(row):
                compound_score = row['sentiment']['compound']
                star_rating = row['star']
                if compound_score >= 0.05:
                    return 'Positive'
                elif compound_score <= -0.05:
                    return 'Negative'
                else:
                    return 'Positive' if star_rating >= 4 else 'Neutral' if star_rating == 3 else 'Negative'
            
            df['sentiment_label'] = df.apply(categorize_sentiment, axis=1)

            st.write("Sentiment Analysis Results")
            st.dataframe(df)

            st.write("Sentiment Distribution")
            sentiment_counts = df['sentiment_label'].value_counts()
            st.bar_chart(sentiment_counts)
        else:
            st.warning("No comments found. Make sure the URL is correct and try again.")
    else:
        st.warning("Please enter a valid URL.")
