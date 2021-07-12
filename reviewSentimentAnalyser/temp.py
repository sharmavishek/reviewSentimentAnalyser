"""
Code Challenge 
Import election_data.csv file
Here is the data of Karnataka Elections for few years. 
Perform analysis based on the problem statement:
Considering the candidates who have contested in more than one Assembly elections,
Do such candidates contest from the same constituency in all the elections? 
If not, does the change of constituency have any effect on the performance of the candidate? 
Considering the candidates who have contested in more than one Assembly elections,
Do such candidates contest under the same party in all the elections? If not,
 how does the change in alliance of the candidate affect the outcome of the next election? 
Do candidates who contested for multiple elections enjoy higher vote share 
percentages compared to the candidates who have contested only once? 
"""


import numpy as np
import pandas as pd
from os import path
from PIL import Image
import pickle
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
df = pd.read_csv("scrapped.csv", index_col=0)

text = " ".join(review for review in df.reviews)
stopwords = set(STOPWORDS)


# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

wordcloud.to_file('j.png')

scrappedReviews = pd.read_csv('scrapped.csv')
    
file = open("my_model.pkl", 'rb') 
pickle_model = pickle.load(file)

file = open("myfeature.pkl", 'rb') 
vocab = pickle.load(file)
def check_review(reviewText):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    reviewText = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(reviewText)

scrap = [check_review(review)[0] for review in scrappedReviews['reviews']]

y = np.array(scrap)
positivelabel = 0
negativelabel = 0
for i in y:
    if i == 0:
        negativelabel += 1
    else:
        positivelabel += 1
z = [positivelabel,negativelabel]
mylabels = ['positive','negative']
plt.pie(z,labels=mylabels,colors=['skyblue','red'])

plt.savefig('pie.jpg')
plt.show() 


