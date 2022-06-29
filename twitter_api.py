from textblob import TextBlob
import tweepy
import configparser
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# read configs
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


# search tweets
keywords = '#onlinelearning'
limit=300

tweets = tweepy.Cursor(api.search_tweets, q=keywords, count=100, tweet_mode='extended').items(limit)

# create DataFrame
columns = ['User', 'Tweet']
data = []

for tweet in tweets:
    data.append([tweet.user.screen_name, tweet.full_text])

    df = pd.DataFrame([tweet.full_text for tweet in tweets], columns=['Tweets'])            ##df = pd.DataFrame(data, columns=columns)

print(df)

##clean the text
##create a function to clean the tweets
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) ##@mentions
    text = re.sub(r'#', '', text) ##removes hastags
    text = re.sub(r'RT[\s]+', '', text)  #removing RT
    text = re.sub(r'https?:\/\/\S+', '', text) ##removing the hyper link
    text = re.sub(r':', '', text)
    return text

df['Tweets']= df['Tweets'].apply(cleanTxt)
##Show clean Text
print(df)


##Creating a function for Subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
    
## Create a function to get ther Polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity
    
    
##Create two new columns
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)

print(df)

## Creating a function for SENTIMENTAL analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score  == 0:
        return 'Neutral'
    else:
        return 'Positive'
    
df['Analysis'] = df['Polarity'].apply(getAnalysis)

#Showing the Dataframe
print(df)

#Plotting the Polarity and Subjectivity
plt.figure(figsize=(8,6))
for i in range(0, df.shape[0]):
    plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='Blue')
    
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

#Showing the value counts
df['Analysis'].value_counts()
plt.title('Sentiments Analysis')
plt.xlabel('Sentiments')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind='bar')
plt.show()