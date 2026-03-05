
# Word2Vec الهدف هنا نتعلم خطّة كاملة لبناء مصنف نصي باستخدام نموذج 
# features ثم استعمال المتوسط لمتجهات الكلمات كمزايا
# RandomForest وتدريب مصنف شجري
# الخطوات قراءة البيانات → تنظيف → تدريب 
# تحويل كل جملة إلى متجه ثابت (متوسط متجهات كلماتها) → تدريب وتصنيف → تقييم الأداء →

import numpy as np
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from nltk import sent_tokenize
import gensim
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from gensim.utils import simple_preprocess

# فيها آلاف التكرارات for تُستخدم أثناء تنفيذ الحلقات الطويلة يعني مثل لما يكون عندك
# يوريك شريط يوضح كم بالمئة انتهى البرنامج وكم باقي، بدل ما تنتظر بصمت tqdm
from tqdm import tqdm  # for i in tqdm(range(10000000)): x=i :مثال
from sklearn.ensemble import RandomForestClassifier


message = pd.read_csv('مسار الملف', encoding='latin1')
print(message.head())

# تنظيف أعمدة غير مفيدة وإعادة تسمية الأعمدة
data = message.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data.rename({
    "v1":"label",
    "v2":"text"}, axis=1, inplace=True)

print(data.head())


lim = WordNetLemmatizer()
corpus= []
for i in range(0,len(data)):
    review= re.sub('[^a-zA-Z]',' ',data['text'][i])
    review= review.lower()
    review= review.split()
    sentance= [lim.lemmatize(word) for word in review] # if not word in stopwords.words('english')]
    review= ' '.join(sentance)
    corpus.append(review)

words= []
for sent in corpus:
    sent_token= sent_tokenize(sent)           # يحول كل نص إلى قائمة جمل
    for sent in sent_token:
        words.append(simple_preprocess(sent)) # يفصل الجملة إلى كلمات صغيرة ومنظفة
print(words)

gensim_model= Word2Vec(words) # بني نموذج الاعدادات الافتراضية
print(gensim_model.wv.most_similar('free'))  # "free" يطبع كلمات قريبة من 

gansim_vocab= gensim_model.wv.key_to_index
print(gansim_vocab)
print(gensim_model.corpus_count)
print(gensim_model.epochs)

print(gensim_model.wv.similar_by_word('bad'))

print(gensim_model.wv['good'].shape) # يعطي شكل المتجه مثلاً (100,)
print(words[0])

# avg_word2vec دالة حساب المتوسط لكل جملة
'''
vocabوتجمع متجه كل كلمة إن وُجد في الـ sentence تأخذ قائمة كلمات 
ترجع صفر OOV ثم تحسب المتوسط إن لم توجد أي كلمة (مثلاً جملة فارغة أو كلها
'''
def avg_word2vec(sentence, model):
    word_vecs= []
    for word in sentence:
        if word in model.wv:
            word_vecs.append(model.wv[word])
    if len(word_vecs) == 0:
        # هذا يعطي تمثيلًا ثابتًا لكل جملة مناسب لمدخلات المصنف
        return np.zeros(model.vector_size) 
    avg_vec= np.mean(word_vecs, axis=0)
    return avg_vec

x= [] # قائمة من المتجهات لكل جملة
# تحويل كل الجمل إلى متجهات وإظهار الأشكال
for i in tqdm(range(len(words))): # تسرع العملية
    x.append(avg_word2vec(words[i], gensim_model)) 
print(x[0])
print(len(x))

# (n_sentences, vector_size) مصفوفة شكلها
xnew= np.array(x)
print(xnew.shape)
print(xnew[0])
print(xnew[0].shape)


y = (data['label'] == 'spam').astype(int).values
y= y.iloc[:,1].values
print(y.shape)

print(x[0].reshape(1,-1).shape)

# train_test_splitمناسب لـ DataFrame تحويل القائمة إلى
df= pd.DataFrame()
df_list= []

for i in range(len(x)):
    df_list.append(pd.DataFrame(x[i].reshape(1,-1)))

df= pd.concat(df_list, ignore_index=True)
print(df.shape)
print(df.head())

x= df
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print(xtrain.head())

# تدريب
classifier= RandomForestClassifier()
classifier.fit(xtrain, ytrain)
predict= classifier.predict(xtest)

print(accuracy_score(ytest, predict))

print(classification_report(ytest, predict))




