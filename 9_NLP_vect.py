# 11) Tf_idf.ipynb
# MLافضل حاجة للـ
# الهدف نعطي وزن لكل كلمة او كلمتين.. بالنسبة للجملة مع الكلمات الثانية


import nltk 
import pandas as pd 
# re.sub('\s+', ' ', text) مثال توحيد المسافات 
import re   # للبحث والتعديل داخل النصوص إزالة الرموز استخراج كلمات استبدال أشياء تنظيف
from nltk.stem import WordNetLemmatizer  # تحويل الكلمات لشكلها القاموسي
from nltk.corpus import stopwords
### TFIDF 
from sklearn.feature_extraction.text import TfidfVectorizer

# UCI غالبًا من spam/hamمجموع بيانات الـ CSV يقرأ ملف
# ضروري لأن الملف يحتوي أحرف/تشفير خاص latin1 باستخدام ترميز
message = pd.read_csv("D:/../spam.csv", encoding='latin1')
message.head()

# يحذف أعمدة غير مفيدة (الأعمدة الفارغة أو الإضافية من الملف الأصلي)
data = message.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)

# اعادة تسمية الاعمدة
data.rename({
    "v1":"label",
    "v2":"text"} ,axis=1,inplace=True)

data.head()

lim= WordNetLemmatizer()  # lemmatization النسخة المحدثة من  
corpus=[] 
for i in range(0,len(data)):
    # [^a-zA-Z] كل ما هو ليس حرف إنجليزي استبدال بفراغ
    # data['text'][i] النص الأصلي للرسالة
    review=re.sub('[^a-zA-Z]',' ',data['text'][i]) 
    review=review.lower()
    review=review.split()
    # setداخل حلقة كثيرًا مكلف؛ من الأفضل تحميلها مرة وحفظها في stopwords ملاحظة: استدعاء
    # if خارج مع stop_words = set(stopwords.words('english')) الافضل 
    sentance=[lim.lemmatize(word) for word in review if not word in stopwords.words("english")]
    review=" ".join(sentance)
    corpus.append(review)

print(corpus)

### TFIDF 
# max_features كم كلمة في كل مرة
tf= TfidfVectorizer(max_features= 100)

# fit_transform يتدرب و يحول
X= tf.fit_transform(corpus).toarray()   # باوزان لاكنها ما زالة تحتوي اصفار array انتاج
print(X)

# ngram_range=(2,2) ياخذ كلمتين كلمتين
tf= TfidfVectorizer(max_features=100,ngram_range=(2,2))
X= tf.fit_transform(corpus).toarray()

print(tf.vocabulary_)
print(X)































