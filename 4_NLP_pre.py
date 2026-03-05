# 6)Stop words.ipynb
# preprocessing الهدف نعمل 
# الهدف نخلي الفقرة بدون الكلمات المميزة في اللغة

from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import WordNetLemmatizer  # نموذج مدرب
## Stop words 
from nltk.corpus import stopwords # الكلمات المميزة في اللغة
import nltk 

# nltk.download('punkt_tab')
# nltk.download('all')  # لتحميل كل المكتبات اذا تريد

## Khalid ibn al-Walid Amazon Text 
paragraph= """
Khalid ibn al-Walid was an undefeated military commander who played a pivotal role in the spread of Islam and the expansion of the Muslim empire. He was a brilliant strategist and tactician, and he was known for his courage, determination, and humility.
Khalid's military achievements were unprecedented. He led the Muslim army to victory over some of the most powerful empires in the world, including the Byzantines and the Persians. His most famous victory was at the Battle of Yarmuk in 636 AD, which marked the beginning of the Muslim conquest of the Middle East.
Khalid was also a deeply religious man. He always gave credit for his victories to Allah. He was also known for his kindness and compassion towards his enemies.
Khalid's story is an inspiration to Muslims and military commanders around the world. He is remembered as a brilliant strategist, a skilled tactician, and a courageous warrior. He is also admired for his humility, his piety, and his dedication to Islam.
"""

print(stopwords.words('english'))
print('-'*100)
print(stopwords.words('arabic'))

stemmer= PorterStemmer()
sentences= sent_tokenize(paragraph)    # نفرط الفقرة لجمل عند النقطة
print('sentences:\n',sentences)
print(type(sentences))

## Apply stopwords then stemming 
for i in range(len(sentences)):
    words= word_tokenize(sentences[i]) # نفرط الجمل لكلمات

    # stopwords يخزن الكلمات اذا ما موجودة من ضمن
    wordss= [stemmer.stem(word) for word in words if word not in set(stopwords.words('english')) ]
    sentences[i]= ' '.join(wordss) ## Converting the word into setences 
print(sentences)

snow= SnowballStemmer('english') 
# تأكد منها ليش مرتين
## Apply stopwords then stemming 
for i in range(len(sentences)):
    words= word_tokenize(sentences[i]) 
    words= [snow.stem(word) for word in words if word not in set(stopwords.words('english')) ]
    sentences[i]= ' '.join(words) ## Converting the word into setences 


print(sentences)

# _______________________نرجع الكلمات لاصلها_____________________
lemmitizaer= WordNetLemmatizer()
for i in range(len(sentences)):
    words= word_tokenize(sentences[i]) 
    words= [lemmitizaer.lemmatize(word.lower(),pos='v') for word in words if word not in set(stopwords.words('english')) ]
    sentences[i]= ' '.join(words) ## Converting the word into setences 


print(sentences)


# ____________لتحسين النتائج POS مع Lemmatize :' '.join(words) طريقة اسهل بدل____________
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

tokens_pos = pos_tag(wordss)  # يعطي قائمة من (word, POS)
# [('cats', 'NNS'), ('are', 'VBP'), ('running', 'VBG'), ('played', 'VBD')] يطلع مثل
'''ملاحظة مهمة: تحتاج لتحميل/نصب مورد 'averaged_perceptron_tagger' أول مرة عبر 
nltk.download('averaged_perceptron_tagger')'''

# هنا يصير حاجة مختلفة شوي (w, get_wordnet_pos(pos))
# تصنيف الكلمة get_wordnet_pos(pos) \ الكلمة w 
# فترجع كل كلمة لاصلها حسب هي ايش تصنيفها
tokens = [lemmitizaer.lemmatize(w, get_wordnet_pos(pos)) for w, pos in tokens_pos]

print(tokens)



