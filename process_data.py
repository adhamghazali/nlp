import nltk
filePath = './data/'#f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re
import string
import numpy as np

from nltk.corpus import twitter_samples
classifier='bayes'

def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean
def build_freqs(tweets, ys):

    yslist = np.squeeze(ys).tolist()

    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs


def count_tweets(result, tweets, ys):


    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in result:
                result[pair] += 1
            else:
                result[pair] = 1
    return result

def prepare_training_data():
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')
    test_pos = all_positive_tweets[4000:]
    train_pos = all_positive_tweets[:4000]
    test_neg = all_negative_tweets[4000:]
    train_neg = all_negative_tweets[:4000]
    train_x = train_pos + train_neg
    test_x = test_pos + test_neg

    if classifier=="lr":
        train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg),1)), axis=0)
        test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)),axis = 0)
    if classifier=='bayes':
        train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
        test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

    if classifier=="lr":
        freqs = build_freqs(train_x, train_y)

    if classifier=='bayes':
        result={}
        freqs = count_tweets(result, train_x, train_y)

    print("type(freqs) = " + str(type(freqs)))
    print("len(freqs) = " + str(len(freqs.keys())))
    return train_x,train_y,test_x,test_y,freqs




