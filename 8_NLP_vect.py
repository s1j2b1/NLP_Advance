# 10) BOW Implementation with n_

import nltk 
import pandas as pd 
import re 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
## Using Count Vectorizer 
from sklearn.feature_extraction.text import CountVectorizer

message = pd.read_csv("مسار الملف", encoding='latin1')
message.head()

data = message.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)
data.head()

names=['label','data'] 

data.rename({
    "v1":"label",
    "v2":"text"} ,axis=1,inplace=True)

data.head()

ps= PorterStemmer()
corpus=[] 

stop_words = set(stopwords.words("english"))
for i in range(0,len(data)):
    review=re.sub('[^a-zA-Z]',' ',data['text'][i]) 
    review=review.lower()
    review=review.split()
    sentance = [ps.stem(w) for w in review if w not in stop_words]
    review=" ".join(sentance)
    corpus.append(review)

# BOWإنشاء الـ 
## Create BOW : For binary set the binary=True 
cv=CountVectorizer(max_features=2500,binary=True) # BOW الأساسي

X=cv.fit_transform(corpus).toarray()
print(X)
print(X.shape)
print(cv.vocabulary_)


## Create BOW : For binary set the binary=True 
# ngram_range نحدد كم كلمة ياخذها من الجملة ليعطيها وزن
cv= CountVectorizer(max_features=2500, binary=True, ngram_range=(1,1)) # n-gramsالآن نجرّب الـ 

X= cv.fit_transform(corpus).toarray()
print(X)
print(cv.vocabulary_)

## Create BOW : For binary set the binary=True 
cv= CountVectorizer(max_features=2500,binary=True,ngram_range=(1,2)) # (1,2)=unigrams + bigrams

X= cv.fit_transform(corpus).toarray()
print(cv.vocabulary_)

## Create BOW : For binary set the binary=True 
cv= CountVectorizer(max_features=2500,binary=True,ngram_range=(2,2)) # (2,2) = bigrams فقط

X= cv.fit_transform(corpus).toarray()
print(cv.vocabulary_)
# (1,3) = unigrams + bigrams + trigrams
cv= CountVectorizer(max_features=2500,binary=True,ngram_range=(1,3))

X= cv.fit_transform(corpus).toarray()
print(cv.vocabulary_)

cv=CountVectorizer(max_features=2500,binary=True,ngram_range=(3,3)) # (3,3) = trigrams فقط

X= cv.fit_transform(corpus).toarray()
print(cv.vocabulary_)


