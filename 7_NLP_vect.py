# 13) one hot encoder.ipynb

# عبارة عن 0 1 مو اوزان array لتحويل الكلمات لارقام؛ لانها تنتج OneHotEncoder ليش ما نستخدم 
# اشان كذا ما نستخدمه
from sklearn.preprocessing import OneHotEncoder # OneHotEncoder أداة لتحويل قيم فئوية إلى تمثيل 
import numpy as np  # للتعامل مع المصفوفات

# نصوص نشتغل عليها كمثال
corpus = [
    "I love AI",
    "AI loves me",
    "I love machine learning"
]

tokens = []

# tokens ثم نجمّع كل الكلمات في قائمة (split) نفصل كل جملة إلى كلمات 
for sentence in corpus:
    words = sentence.split()
    tokens.extend(words)

# Step 2: Make vocabulary (unique words)
# indicesلنتحكّم في ترتيب الـ sorted
vocab = sorted(set(tokens))  # اشان ما تتكرر الكلمات set
print("Vocabulary:\n", vocab)

# Step 3: Reshape for sklearn OneHotEncoder
# fitتحويل قائمة الكلمات إلى مصفوفة شكلها عمودي هذا مطلوب للـ
words_array = np.array(vocab).reshape(-1, 1)
print('words_array\n',words_array)


# Step 4: Initialize encoder
encoder = OneHotEncoder()

# Fit the encoder on vocabulary
# يتعلم على القائمة بحيث يربط كل كلمة بعمود واحد في التمثيل داخليًا يصنع خريطة 
# { 'AI':0, 'I':1, 'love':2, ... }
x= encoder.fit(words_array)
print('x\n',x)


# Step 5: Encode each sentence
'''
نطبع كل كلمة مع المتجه المقابل (أصفار وواحدات)
[0 0 1 0 0 0 0] love مثال لمتجه كلمة 
:ليش المتجه عبارة عن 0 و1 مش أوزان
كل عمود يرمز لكلمة واحدة فقط لذا قيمة العمود المقابل للكلمة=1 والباقي=0
هذا ليس تمثيلاً دلاليًّا؛ هو مجرد تعريف فريد لكل كلمة لا يحمل أي معلومات عن التشابه أو المعنى
'''
for i, sentence in enumerate(corpus):
    words = sentence.split()
    encoded = encoder.transform(np.array(words).reshape(-1, 1))
    print(f"\nSentence {i+1}: '{sentence}'")
    for word, vector in zip(words, encoded):
        print(f"{word:10s} → {vector.astype(int)}")   # ربما يعطي مصفوفة صغيرة صحيحة
        print(f"{word:10s} → {vector.toarray()[i]}")  # لطباعة مصفوفة كاملة أنظف استخدم



'''لماذا لا نستخدم One-Hot عادةً في NLP الحديث؟
اذا لديك 100 كلمة المتجه طوله 100 لكل كلمة vocabularyطول المتجه حجم الـ
هذا ضخم جدًا في الذاكرة والحساب و لا يوجد تشابه معنوي 
سيكونان متجهين متباعدين تمامًا cat و dog
كل كلمة جديدة تحتاج عمودًا جديدًا OOV مشكلة الكلمات النادرة و 
'''




















