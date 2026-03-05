# 7)Part Of Speech Tagging.ipynb
# preprocessing الهدف نعمل 
# الهدف نجيب تصنيف كل شي داخل الفقرة

from nltk import WordNetLemmatizer
from nltk import word_tokenize
## Stop words 
from nltk.corpus import stopwords 
import nltk 
# nltk.download('maxent_ne_chunker_tab')

# الانواع او التصنيفات
### Nouns
# - **NN**: Noun, singular or mass (e.g., "dog", "car")
# - **NNS**: Noun, plural (e.g., "dogs", "cars")
# - **NNP**: Proper noun, singular (e.g., "John", "London")
# - **NNPS**: Proper noun, plural (e.g., "Americans", "Sundays")

# ### Pronouns
# - **PRP**: Personal pronoun (e.g., "I", "he", "she", "we")
# - **PRP$**: Possessive pronoun (e.g., "my", "your", "his", "her")
# - **WP**: Wh-pronoun (e.g., "what", "who", "whom")
# - **WP$**: Possessive wh-pronoun (e.g., "whose")

# ### Verbs
# - **VB**: Verb, base form (e.g., "run", "go")
# - **VBD**: Verb, past tense (e.g., "ran", "went")
# - **VBG**: Verb, gerund or present participle (e.g., "running", "going")
# - **VBN**: Verb, past participle (e.g., "run", "gone")
# - **VBP**: Verb, non-3rd person singular present (e.g., "run", "go")
# - **VBZ**: Verb, 3rd person singular present (e.g., "runs", "goes")

# ### Adjectives
# - **JJ**: Adjective (e.g., "big", "blue")
# - **JJR**: Adjective, comparative (e.g., "bigger", "bluer")
# - **JJS**: Adjective, superlative (e.g., "biggest", "bluest")

# ### Adverbs
# - **RB**: Adverb (e.g., "quickly", "never")
# - **RBR**: Adverb, comparative (e.g., "faster", "better")
# - **RBS**: Adverb, superlative (e.g., "fastest", "best")
# - **WRB**: Wh-adverb (e.g., "how", "where", "when")

# ### Determiners
# - **DT**: Determiner (e.g., "the", "a", "this")
# - **PDT**: Predeterminer (e.g., "all", "both")
# - **WDT**: Wh-determiner (e.g., "which", "that")

# ### Conjunctions
# - **CC**: Coordinating conjunction (e.g., "and", "but", "or")
# - **IN**: Preposition or subordinating conjunction (e.g., "in", "of", "like", "because")

# ### Other
# - **MD**: Modal (e.g., "can", "will", "should")
# - **CD**: Cardinal number (e.g., "one", "two", "100")
# - **EX**: Existential "there" (e.g., "there is")
# - **FW**: Foreign word (e.g., "bonjour", "faux")
# - **LS**: List item marker (e.g., "1", "2", "3")
# - **POS**: Possessive ending (e.g., "'s")
# - **RP**: Particle (e.g., "off", "up", "out")
# - **SYM**: Symbol (e.g., "$", "%", "#")
# - **TO**: "to" as a preposition or infinitive marker (e.g., "to go", "to the store")
# - **UH**: Interjection (e.g., "uh", "oh", "wow")

# ### Punctuation
# - **.**: Sentence-final punctuation (e.g., ".", "?", "!")
# - **,**: Comma (e.g., ",")
# - **:**: Colon or ellipsis (e.g., ":", "...")
# - **(`)**: Opening parenthesis (e.g., "(", "[", "{")
# - **)**: Closing parenthesis (e.g., ")", "]", "}")

paragraph= """
Khalid ibn al-Walid was an undefeated military commander who played a pivotal role in the spread of Islam and the expansion of the Muslim empire. He was a brilliant strategist and tactician, and he was known for his courage, determination, and humility.
Khalid's military achievements were unprecedented. He led the Muslim army to victory over some of the most powerful empires in the world, including the Byzantines and the Persians. His most famous victory was at the Battle of Yarmuk in 636 AD, which marked the beginning of the Muslim conquest of the Middle East.
Khalid was also a deeply religious man. He always gave credit for his victories to Allah. He was also known for his kindness and compassion towards his enemies.
Khalid's story is an inspiration to Muslims and military commanders around the world. He is remembered as a brilliant strategist, a skilled tactician, and a courageous warrior. He is also admired for his humility, his piety, and his dedication to Islam.
"""

sentences= word_tokenize(paragraph)  # نفرط الجملة لكلمات
print(sentences)

# طريقة ثانية نفرط لكلمات مع نوع الكلمة و استبعاد الكلمات الشائعة
for i in range(len(sentences)):
    words= word_tokenize(sentences[i])  # نفرط الجمل لكلمات
    # stopwords يخزن الكلمات اذا ما موجودة من ضمن
    words= [word for word in words if word not in set(stopwords.words('english')) ]
    pos_tag= nltk.pos_tag(words)  # يضيف على الكلمة الي طلعت النوع
    print(pos_tag)

# طريقة ثانية نفرط الكلمات و نضيف النوع
umar= "Khalid ibn al-Walid was an undefeated military"
umar= word_tokenize(umar)                    # يفرط لكلمات
umar= [umar]

tagged_sentences = nltk.pos_tag_sents(umar)  # يضيف النوع لكلك كلمة
print(tagged_sentences)

tag_element= nltk.pos_tag(words)   # يضيف النوع لكل كلمة اذا تريد ترسم
print(tag_element)
nltk.ne_chunk(tag_element).draw()  # عرض كل كلمة و تصنيفها بالرسم

# طريقة غيرها نفرط لكلمات
mysp= "Khalid ibn al-Walid was an undefeated military".split()
print(mysp)

tagged_sentences = nltk.pos_tag_sents(mysp)
print(tagged_sentences)





