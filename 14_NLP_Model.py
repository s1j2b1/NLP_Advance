
# الهدف بناء نموذج يفهم سياق الجملة و يصنف الكلمات
# نعرف الكلمات القريبة بالمعنى 
# نحفظ و نحمل النموذج

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess # تفرط و تعمل معالجة للنص رموز ما رموز

corpus = ["the cat, sa't on &the mat", "the dog! sat on the mat"]

# نفرط و ننظف النص
# Word2Vecتنشأ قائمة مناسبة للـ simple_preprocess
text = [simple_preprocess(doc) for doc in corpus]
print(text)

# Word2Vec بناء النموذج
model = Word2Vec(
    sentences=text,        # قائمة النص المعالج
    vector_size=100,       # أبعاد المتجه لكل كلمة (dimensionality)
    window=5,              # ينظر إلى 5 كلمات قبل وبعد Word2Vec يعني
    min_count=1,           # 1 تجاهل الكلمات التي تظهر أقل من - (vocabulary) يقلل الضوضاء ويصغّر قاموس الكلمات   
    workers=4,             # يسرّع العملية على معالجات متعددة
    sg=1,                  # يحاول يعرف السياق من الكلمة skip-gram = 1 \ 0 = CBOW يحاول يعرف الكلمة من السياق
    epochs=10              
)

# cosine similarity يقوم بحساب التشابه
# النتيجة: رقم بين -1 و 1. قيمة قريبة من 1 → المتجهان متشابهان؛ قرب 0 → غير مرتبطين؛ سالب → متعاكسين.
sim = model.wv.similarity('cat', 'dog')
print("similarity(cat, dog) =", sim)

# أكثر الكلمات تشابهًا مع كلمة معينة طبعا على حسب سياق النصوص المدرب عليها
# 'cat' مع أعلى 10 كلمات قريبة من (word, score) يرجع؟: قائمة من أزواج
print("most similar to 'cat':", model.wv.most_similar('cat', topn=10))

# ثم يعيد أقرب الكلمات للمتجه الناتج vec('king') - vec('man') + vec('woman') يحسب المتجه
print("king - man + woman ->", model.wv.most_similar(positive=['king','woman'], negative=['man'], topn=5))

# حفظ النموذج
model.save("word2vec_spam_model.model")

# ثم لاحقًا: تحميل النموذج
from gensim.models import Word2Vec
model = Word2Vec.load("word2vec_spam_model.model")














































