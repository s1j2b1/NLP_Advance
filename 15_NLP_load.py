# الهدف 

### Import the Libraries
import numpy as np 
from keras.datasets import imdb          # داتا لمراجعات أفلام
from keras.preprocessing import sequence # لتوحيد طول الجمل
from keras.layers import Embedding , SimpleRNN ,Dense # بسيطة، وطبقة إخراج RNN طبقة تمثيل كلمات، طبقة
from keras.models import load_model      # لتحميل موديل محفوظ

import warnings 
warnings.filterwarnings('ignore')        # في الإخراج Pythonإخفاء تحذيرات الـ



### Load The IMDB dataset word Index 
# (word index) جلب قاموس الكلمات
'''تبدأ عادةً من 1 word_index ملاحظة قيم
هناك إزاحة عند تحميل البيانات لأن بعض المؤشرات محجوزة لحالات خاصة IMDBلكن في الــ
لهذا في فك الترميز سيُطرح 3 padding, start, unknown مثل 
'''
word_index= imdb.get_word_index()  # dict يرجع 
# تبادُل المفاتيح والقيم ليصبح رقم → كلمة، وهذا يستخدم لاحقًا لفك ترميز الجملة المشفّرة إلى نص مفهوم
reverse_word_index= { value:key for key,value in word_index .items()}


### Load my per_Trained Model 
# تحميل الموديل المحفوظ وإظهار ملخصه
try:
    model = load_model('simple_rnn_imdb.h5')
    print(model.summary())     # parametersيطبع بنية الشبكة الطبقات وأحجام الـ
    print(model.get_weights()) # يرجع قائمة المصفوفات (الأوزان)
except AttributeError as e:  # هو نوع من الأخطاء في بايثون يظهر لما تحاول تستخدم خاصية أو دالة غير موجودة على كائن
    print(f"Error loading model: {e}")

# WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.

'''
# Embeddingطبقة الـ 
# هذا الشكل يوضح أبعاد البيانات وهي تمر عبر الطبقات العصبية Output Shape
# يعني: عندنا 32 جملة، كل جملة فيها 500 كلمة، وكل كلمة تحولت إلى متجه رقمي من 128 قيمة
# يعني عندنا 10,000 كلمة في المفردات كل كلمة تمثلها 128 رقم يمكن النموذج أن يتعلمها Param
# 1,280,000 = 10,000 * 128
# تتعامل مع التسلسلات وتتعلم العلاقات الزمنية بين الكلمات SimpleRNNطبقة الـ 
# عدد المعاملات = units * (units + input_dim + 1) عدد معلماتها يحسب بهذه الصيغة
# RNN عدد الخلايا في units = 128
# Embeddingلأنها تستقبل متجه من طبقة الـ input_dim = 128 
# (انحياز) bias لأن كل خلية فيها bias +1
# 128 * (128 + 128 + 1) = 32,896
# الإخراج هي الطبقة الأخيرة التي تعطي النتيجة (إيجابي أو سلبي) Denseطبقة الـ
# عدد المعاملات = عدد المدخلات * عدد المخرجات + bias ← 128 * 1 + 1 = 129
#  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
# │ embedding_1 (Embedding)         │ (32, 500, 128)         │     1,280,000 │ 
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ simple_rnn_1 (SimpleRNN)        │ (32, 128)              │        32,896 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense_1 (Dense)                 │ (32, 1)                │           129 │
# └─────────────────────────────────┴────────────────────────┴───────────────┘
'''
print(model.get_weights())

## Step 2 
# دوال المعالجة وفك التشفير
# الهدف تحويل قائمة أرقام مثل [1,14,20,...] إلى كلمات مفهومة
def decode_review(encoded_review):
    # يتم إضافة إزاحة +3 IMDB نطرح 3 لاسترجاع المفتاح الصحيح لأن عند تخزين بيانات
    # ? يعني إذا الرقم غير موجود يرجع get(i-3, '?')
    return " ".join([reverse_word_index.get(i-3,'?') for i in encoded_review])


def preprocessing_text(text):
    words= text.lower().split()
    # إذا لم يكن موجودًا يعيد 2 IMDB يحصل رقم الكلمة من قاموس word_index.get(word,2)
    # + 3 لإعادة التطبيق لنفس إزاحة  
    encoded_review= [word_index.get(word,2) + 3 for word in words]
    # ليصبح طوله 500 (النموذج مدرّب على 500 طول). الناتج شكل (1,500)
    # الأكواد تضيف إزاحة عند تحويل الكلمات إلى أرقام Keras IMDB لمَ نضيف +3؟ لأن  
    padded_review= sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

# ---------------------------------------------------------------

### Prediction Function 
# دالة التنبؤ
def predict_sentiment(review):
   preprocessed_input= preprocessing_text(review)  # نجهّز النص
   prediction= model.predict(preprocessed_input)    # [[0.86]] يعطي مصفوفة احتمالات مثلاً
   sentiment= 'Positive' if prediction[0][0] > 0.5 else 'Negative' # نحدد التصنيف: إذا الاحتمال > 0.5 نعتبره إيجابي
   return sentiment , prediction[0][0] 

### User input
### Take An Examples 
# مثال تشغيل يطبّق الدالة على جملة اختبار ويطبع النتيجة والاحتمال
example_review= "The Movie was fantastic! The Acting was great and the plot was thrilling"
sentiment,score= predict_sentiment(example_review)

print(f'Review: {example_review}')
print(f'Sentiment: {sentiment}')
print(f'prediction Score: {score}')


'''Sentiment Analysis الهدف يريك خط أنابيب عملي كامل لتطبيق تصنيف المشاعر
(IMDB) مجموع بيانات جاهزة
لتمثيل الكلمات رقمياً Embedding طبقة
بسيطة لمعالجة التسلسلات الزمنية RNN شبكة
طريقة تحميل نموذج جاهز واختباره على جملة جديدة.
'''

