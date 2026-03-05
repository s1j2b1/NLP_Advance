# 9) BOW Implementation.ipynb

import nltk 
import pandas as pd 
import re 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
## Using Count Vectorizer 
from sklearn.feature_extraction.text import CountVectorizer


message = pd.read_csv("مسار الملف", encoding='latin1')
print(message.head())

data = message.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)
print(data.head())

data.rename({
    "v1":"label",
    "v2":"text"} ,axis= 1, inplace= True)

print(data.head())

ps= PorterStemmer()  # لتحويل الكلمات إلى جذورها
corpus= [] 

stop_words = set(stopwords.words("english"))
for i in range(0,len(data)):
    # استبدال كل ما ليس حرفاً بمسافة (إزالة أرقام/رموز)
    review=re.sub('[^a-zA-Z]',' ',data['text'][i]) 
    review=review.lower()
    review=review.split()
    sentance = [ps.stem(w) for w in review if w not in stop_words]
    review=" ".join(sentance)
    corpus.append(review)

## Create BOW : For binary set the binary=True 
# استبعاد الكلمات المتكررة اكثر من 2500 max_features
# بدلاً من تخزين عدد مرات ظهور الكلمة binary=True
# يخزن 1 إذا الكلمة موجودة في المستند، و0 إذا غير موجودة
# وليس تكرارها free,win لأن في بعض مشروعات أهم شيء هو وجود كلمة معبرة عن السبام مثل
cv= CountVectorizer(max_features=2500, binary=True)  # binary=False جرب 

# sparse matrixيتدرب و يحول كل مستند إلى صف من الـ
X= cv.fit_transform(corpus).toarray()
X1= cv.fit_transform(corpus)  # keep sparse

print(X)
print(X.shape)

print(X1)
print(X1.shape)

