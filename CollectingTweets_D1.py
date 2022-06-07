#import the necessary methods from tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import re

#variables that contains the user credential to access twitter API
access_token = "1272349540717170690-XE4HOf9ijdGshg4NnJkjzMAy7UicyA"  
access_token_secret = "J9nI4blloXw4QZgQOv13aMdRKWIJrbqOWJOoa7A4BWRNJ"  
consumer_key = "JTEKxpfNlfeQaxaflxJMUoEbV"  
consumer_secret = "HWxKzyFlOtdYj3MTHU6MgQShrk6NILsgZMT3fGcoJQXUa2d1eN"  

#create tracklist with the words that will be searched for
tracklist = ['*']

#initialize Global variable
tweet_count = 0

#input number of tweets to be downloaded
n_tweets = 1000
f = open("D1.txt", "w")
f.close()

#create the class that will handle the tweet stream
class StdOutListener(StreamListener):
         def on_data(self, data):
             global tweet_count
             global n_tweets
             global stream
             #tweet={}
             if tweet_count < n_tweets:
                 try:
                     print(tweet_count,data,"\n")
                     tweet_data = json.loads(data)
                     pattern1 = re.compile(r'\n')
                     tweet_txt = pattern1.sub(r'', tweet_data['text'])
                     pattern2 = re.compile(r'RT')
                     tweet = pattern2.sub(r'', tweet_txt) 
                     f=open("/Users/kenngai/Desktop/Asn2/D1.txt", "a+")
                     f.write(str(tweet_data['id']) + "\t" + tweet + "\n")   
                     tweet_count += 1
                        
                 except BaseException:
                     print("Error:", tweet_count, data)

                 return True
             else:
                 stream.disconnet()
         
         def on_error(self,status):
             print(status)
 
# Handles twtter authetification and the connection to twitter
l = StdOutListener()
auth = OAuthHandler(consumer_key, consumer_secret)  
auth.set_access_token(access_token, access_token_secret)  
stream = Stream(auth,l)



#stream without query search and stream - D1
#stream.sample(is_async=True)

#stream with query search and stream - D2
stream.filter(track=tracklist,languages=["en"])