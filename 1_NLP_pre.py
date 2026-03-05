# 3) Tokenization Example.ipynb
# للفقرة بفرط الكلمات preprocessing الهدف نعمل 

import nltk 
#Tokenization
# ============================ 
from nltk.tokenize import sent_tokenize  # نفرط الفقرة لجممل
from nltk import word_tokenize           # نفرط الجمل لكلمات
from nltk import wordpunct_tokenize      # نفرط الجمل لكلمات ناد الاستخدام
from nltk.tokenize import TreebankWordDetokenizer # نفرط لكلمات شكل مختلف
# nltk.download('punkt_tab')  # مكتبة داخلية ما تنزل الا كذا مرة وحدة

corpus= """Hello Welcome , To Mohammad NLp Tutorilas.
Please Do watch The Entire course! To Become Expert In Nlp .
"""

print(corpus)
documents= sent_tokenize(corpus)  # نفرط الفقرة لجمل
type(documents)

for i in documents:
    print(i)

## Tokenization 
## Paragraph --> words 
## Sentence ---> words 

print(word_tokenize(corpus))         # نفرط الجمل لكلمات

for i in documents:                  # forنفرط الجمل لكلمات بالـ
    print(word_tokenize(i))

print(wordpunct_tokenize(corpus))    # فنكشن ثاني ممكن نفرط الجمل لكلمات نادرة الاستخدام

tokenizer= TreebankWordDetokenizer() # نفرط الجمل لكلمات بشكل مختلف
tokenizer.tokenize(corpus)













