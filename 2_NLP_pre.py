# 4)Stemming.ipynb
# preprocessing الهدف نعمل 
# نرجع الكلمات لاصلها و نختار اللغة

import nltk 
from nltk.stem import PorterStemmer     # يرجع الكلمات لاصلها
from nltk.stem import RegexpStemmer     # نحدد الاشياء الي يشيلها
from nltk.stem import SnowballStemmer   # نحدد اللغة

words= ["eating","eats","eaten","writing","writes","programming","programs","history","finally","finalized"]

stemmer= PorterStemmer()
for word in words:
    print(word + "-------->" + stemmer.stem(word))  # ارجاع الكلمات لاصلها

print(stemmer.stem("sitting"))        # نرجع كلمات معينة لاصلها

# نحدد الاشياء الي يشيلها
reg_stemmer= RegexpStemmer('ing$|s$|e$|able$', min=4) 
print(reg_stemmer.stem("eating"))     # نشيل الحروف المحددة من كلمات معينة
print(reg_stemmer.stem("ingeating"))

# أحيانًا يطلع كلمات ما لها معنى Stemmer الـ
# goose ✅ <-gees❌<-Geese :لا تفهم نوع الكلمة ولا تستخدم قاموس لغوي مثال
snowball= SnowballStemmer("english")  # نحدد اللغة

for word in words:
    print(word + "-------->" + snowball.stem(word))

print(stemmer.stem('fairly') , stemmer.stem('sportingly') )
print(snowball.stem('fairly') , snowball.stem('sportingly'))
print(snowball.stem('going'))
print(snowball.stem('goes'))


print('مرحبا')

def arab_txt(text):  # لاظهار الكتابة العربية بشكل صحيح
    import arabic_reshaper
    from bidi.algorithm import get_display
    reshaped = arabic_reshaper.reshape(text) # يرتّب الحروف داخل الكلمة (shaping)  
    bidi_text = get_display(reshaped)        # يعدّل الاتجاه للعرض في بيئة LTR
    return bidi_text

snowball= SnowballStemmer("arabic")  # نحدد اللغة
print(arab_txt(snowball.stem('القمر')))





