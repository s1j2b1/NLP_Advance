# (Spam Detector) بناء موديل كشف الرسائل المزعجة 


import pandas as pd   # وجداول البيانات CSV للتعامل مع ملفات
import re             # تنظيف النص من الرموز والأرقام regexللـ 
from nltk.stem import WordNetLemmatizer  # للأدوات اللغوية
from nltk.corpus import stopwords        # كلمات التوقف
from sklearn.feature_extraction.text import TfidfVectorizer # لتحويل النص إلى أرقام
from sklearn.feature_extraction.text import CountVectorizer # لتحويل النص إلى أرقام
from sklearn.model_selection import train_test_split        # لتقسيم البيانات تدريب/اختبار
from sklearn.naive_bayes import MultinomialNB   # خوارزمية بسيطة وفعّالة لتصنيف النصوص
from sklearn.metrics import accuracy_score, classification_report  # لتقييم الموديل

# latin1 الكمبيوتر كيف يفسر الحروف
message = pd.read_csv('D:/Nlp...', encoding='latin1')
print(message.head())

# حذف أعمدة فارغة/غير مرغوبة
data = message.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# اعادة تسمية الاعمدة
data.rename({
    "v1":"label",
    "v2":"text"}, axis=1, inplace=True)

print(data.head())

lim = WordNetLemmatizer()
corpus= []

stop_words = set(stopwords.words('english'))
for i in range(0,len(data)):
    # يستبدل أي شيء ليس حرفًا إنجليزيًا بمسافة
    review= re.sub('[^a-zA-Z]',' ',data['text'][i]) 
    review= review.lower()
    review= review.split()
    # لكل كلمة lemmatize ثم يطبق stopwords يستبعد كلمات التوقف
    sentance= [lim.lemmatize(w) for w in review if w not in stop_words]
    review= ' '.join(sentance)  # يعيد كلمات الجملة كسلسلة واحدة
    corpus.append(review)

print(corpus)

# y (label → 0/1) تجهيز المتغير الهدف
y = (data['label'] == 'spam').astype(int).values
y= y.iloc[:,1].values  # spam = 1 يحول العمود الثاني الى مصفوفة غالبًا يمثل
print(y)

# للحفاظ على نفس توزيع الفئات في التدريب والاختبار stratify=y
xtrain, xtest, ytrain, ytest = train_test_split(corpus,y, test_size=0.2, random_state=42, stratify=y)

# 1 → كلمة واحدة (Unigram)
# 2 → كلمتان متجاورتان (Bigram)
# 3 → ثلاث كلمات (Trigram)
# الكلمات المكررة اكثر من 2500 فقط max_features
cv= CountVectorizer(max_features=2500, ngram_range=(1,2))

# # يتعلّم الكلمات (يبني القاموس) ويحوّل النصوص إلى أرقام
xtrain= cv.fit_transform(xtrain).toarray()  
# يستخدم القاموس اللي تعلمه من قبل فقط لتحويل بيانات جديدة (ما يعيد التعلم)
xtest= cv.transform(xtest).toarray()

print(cv.vocabulary_)  # يطبع map الكلمة
print(xtrain)          # لمراجعة الأشكال والقيم
print(y)

# التدريب
spam_detect_model= MultinomialNB()
spam_detect_model.fit(xtrain,ytrain)

# الاختبار
predict= spam_detect_model.predict(xtest)

# التقييم
print(accuracy_score(ytest, predict))
print(classification_report(ytest, predict))


# ------------------------------------------------------------------------
# TfidfVectorizerبـ CountVectorizer نعيد الكود بنفس الخطوات لكن نستبدل

xtrain, ytrain, xtest, ytest = train_test_split(corpus,y, test_size=0.2, random_state=42)

cv= TfidfVectorizer(max_features=2500, ngram_range=(1,2))
xtrain= cv.fit_transform(xtrain).toarray()
xtest= cv.transform(xtest).toarray()

# التدريب
spam_detect_model= MultinomialNB()
spam_detect_model.fit(xtrain,ytrain)

# الاختبار
predict= spam_detect_model.predict(xtest)
print(accuracy_score(ytest, predict))

# التقييم
print(classification_report(ytest, predict))




