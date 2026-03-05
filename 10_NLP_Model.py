# الهدف نستخدم كود لنموذج مدرب خاص بأحد الشركات

# Word2Vec, FastText مشهورة لمعالجة الكلمات والمتجهات وتدريب نماذج مثل 
import gensim

# ..او Word2Vecخفيف للبحث وليس للتدريب؛ إذا أردت تدريب أو تحديث الموديل ستستخدم KeyedVectors
from gensim.models import KeyedVectors
import gensim.downloader as api  # لتحميل موديلات ومعاجم مدرّبة مسبقًا

# قبل التحميل، استعمل السطر التالي اذا تريد حفظة في مكان محدد
api.BASE_DIR = "D:\Viewer\JORDAN\Ai\Ai_Models"   # ← هذا مجلد الفلاش مثلاً

# Google News مدرّب على Word2Vec يحمل نموذج
wv= api.load('word2vec-google-news-300')  # رمز النموذج المدرب من جوجل 

# KeyedVectors نسخة خفيفة من المتجهات فقط word2vec-google-news-300.kv الملف الناتج
wv.save("D:\Viewer\JORDAN\Ai\Ai_Models/word2vec-google-news-300.kv")  # بعد ما ينتهي التحميل

# بعد فترة، مثلاً في لابتوب ثاني أو من دون إنترنت
wv = KeyedVectors.load("D:\Viewer\JORDAN\Ai\Ai_Models/word2vec-google-news-300.kv", mmap='r')

# ملاحظة إضافية إذا أردت تحديد مجلد التخزين الافتراضي ⚡
# set GENSDATA_DIR= D:\Viewer\JORDAN\Ai\Ai_Models  # بحيث كل النماذج القادمة تنزل هناك تلقائيًا.

# KeyError سيحدث vocabملاحظة: إذا الكلمة غير موجودة في الـ
vec_king= wv['king']  # 'king' متجه الكلمة 

print('vec_king\n',vec_king)              # القيم العددية في المتجه
print('vec_king.shape\n',vec_king.shape)  # (300,) شكل المصفوفة، غالبًا
print('wv king\n',wv['queen'])            # queen متجه كلمة

print(wv.most_similar('football'))       # football الكلمات المصنفة ضمن
print(wv.most_similar('happy'))
print(wv.similarity("hockey","sports"))  # نسبة التقاررب بين كلمتين

# (queen) يعني بالعقل : 👑 الملك رجل إذا أزلنا "رجل" وأضفنا "امرأة" الناتج المنطقي لازم يكون ملكة
# king → queen ليش نطرح ونضيف متجهات؟ عشان نحاكي العلاقات بين المعاني مثل 🧮 
vec= wv['king']-wv['man']+wv['woman']
print(vec)

print(wv.most_similar([vec]))  # أو ما يماثلها 'queen' الهدف إيجاد 

# 
word= wv['xyzword']
if word in wv.key_to_index:  # قاموس داخلي فيه كل الكلمات المعروفة
    print(wv[word])
else:
    print("الكلمة غير موجودة في الموديل")

# أو تستخدم الطريقة الثانية (نفس الفكرة لكن أوضح)
if wv.has_index_for(word):   # دالة مخصصة تفحص وجود الكلمة بطريقة أنظف
    print(wv[word])









