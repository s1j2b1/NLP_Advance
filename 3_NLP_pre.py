# 5)Lemmatization.ipynb
# preprocessing الهدف نعمل 
# نرجع الكلمات لاصلها مع اختيار نوع الكلمات
# استخدام نموذج مدرب لذالك

import tensorflow
import nltk
nltk.download('punkt_tab')
# نحدد طبيعة الكلمات الي نريدة يرجهن لاصلهن هل افعال او اسماء او صفات
from nltk import WordNetLemmatizer # نموذج مدرب

# من الإنترنت إلى جهازك محليًا ينزل لمرة واحدة NLTK هذان السطران يحمّلان موارد
# nltk.download('stopwords') # قائمة الكلمات التوقّف
# nltk.download('wordnet')   # قاعدة بيانات معجمية تُستخدم من قبل WordNetLemmatizer

words= ["eating","eats","eaten","writing","writes","programming","programs","history","finally","finalized"]

lemma= WordNetLemmatizer()
for word in words:
    print(f"old:{word} , new:{lemma.lemmatize(word,pos='n')}") 

print(lemma.lemmatize("going",pos="v"))
print(lemma.lemmatize("going",pos="a"))
print(lemma.lemmatize("goes",pos="a"))
print(lemma.lemmatize('fairly') , lemma.lemmatize('sportingly') )


# ______________لتحسين النتائج POS مع Lemmatize :مثال_______________
from nltk import pos_tag
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): # (adjective) بداية وسم الصفات 'J'
        return wordnet.ADJ  # يرجع التصنيف للكلمة
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # كخيار افتراضي NOUN إذا الوسم لم يطابق أي قائمة أعلاه، نُعيد
        return wordnet.NOUN

tokens_pos = pos_tag(words)  # يعطي قائمة من (word, POS)
# [('cats', 'NNS'), ('are', 'VBP'), ('running', 'VBG'), ('played', 'VBD')] يطلع مثل
'''ملاحظة مهمة: تحتاج لتحميل/نصب مورد 'averaged_perceptron_tagger' أول مرة عبر 
nltk.download('averaged_perceptron_tagger')'''

# هنا يصير حاجة مختلفة شوي (w, get_wordnet_pos(pos))
# تصنيف الكلمة get_wordnet_pos(pos) \ الكلمة w 
tokens = [lemma.lemmatize(w, get_wordnet_pos(pos)) for w, pos in tokens_pos]


print(tokens)












