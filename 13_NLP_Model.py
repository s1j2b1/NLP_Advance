# IMDB خطّة كاملة لبناء نموذج تصنيف مشاعر بسيط على بيانات 


import os
for dirname,_, filenames in os.walk('مسار المشروع'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

### Import the Libraries
from keras.datasets import imdb
from keras.preprocessing import sequence 
from keras.models import Sequential 
from keras.layers import Embedding , SimpleRNN ,Dense  # طبقات الشبكة
import warnings 
warnings.filterwarnings('ignore')  # يخفي التحذيرات
 
from keras.models import load_model

# اقتصار المفردات على أكثر 10000 كلمة تكرارًا max_features
max_features= 10000
(xtrain,ytrain),(xtest,ytest) = imdb.load_data(num_words= max_features)

# paddingصحيح لكن بعد الـ 
# هذا سطر توضيحي — لكن لا تعمل .shape الآن لأن xtrain قائمة
print(f'train data shep: {xtrain.shape}, train label shape: {ytrain.shape}')
print(f'train data shep: {xtest.shape}, train label shape: {ytest.shape}')

# هذه الأعداد هي مؤشرات كلمات، ليست نصًا مفهوماً بعد.
print(xtrain[0])
sample_review= xtrain[0]
sample_label= ytrain[0]
print(f'sample_review as integers: {sample_review}')
print(f'sample_label: {sample_label}')


word_index= imdb.get_word_index() # word -> index يعيد قاموسًا
# index -> word يعكسه لتمكنك من تحويل الأرقام إلى كلمات reverse_word_index 
reverse_word_index= {value:key for key,value in word_index.items()}

# تمسح أو تحجز المؤشرات 0,1,2 لاغراض خاصة لذلك لإيجاد الكلمة الصحيحة نطرح 3Keras لأن مكتبة
# يرجع ? إذا لم نجد الكلمة
decoded_review= ' '.join([reverse_word_index.get(i-3,'?') for i in sample_review]) 

max_len= 500
# إما بقص الأطول أو حشو الأقصر بالأصفار max_len يجعل كل تتابع طوله pad_sequences
# num_samples, max_len يصبحان مصفوفات كشكلxtrain وxtest بعد هذا السطر
xtrain= sequence.pad_sequences(xtrain, maxlen= max_len)
xtest= sequence.pad_sequences(xtest, maxlen= max_len)
print(xtrain)

# index حجم المفردات الأقصى max_features
# 128 = طول متجه كل كلمة
model = Sequential([
    Embedding(max_features, 128, input_length= max_len),
    SimpleRNN(128, activation='tanh'),   # عادة SimpleRNN يستخدم tanh
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
erlystop= EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

histor= model.fit(
    xtrain,ytrain,
    epochs=100,
    batch_size=32,
    validation_data=(xtest,ytest),
    callbacks=[erlystop],
    verbose=1
    )




