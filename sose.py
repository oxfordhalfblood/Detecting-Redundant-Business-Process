# SOSE Project Code

from nltk.stem.wordnet import WordNetLemmatizer
from numpy import argmax
import pandas as pd
import numpy as np
import nltk
import sklearn
from nltk import RegexpParser
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer

h1 = "When the purchasing department would write a purchase order, they sent a copy to accounts payable. Then, the material control would receive the goods,and send a copy of the related document to accounts payable. At the same time, the vendor would send a receipt for the goods to accounts payable"

# h2 = "Then, the material control would receive the goods, and send a copy of the related document to accounts payable"
# h3 = "At the same time, the vendor would send a receipt for the goods to accounts payable"


# news_headlines = [news_headline1, news_headline2,
#                   news_headline3, news_headline4]
# tokenizer = RegexpTokenizer(r'\w+')
# tokenizer.tokenize(h1)

news_headlines = [h1]
# Tokenize headline to list of words
news_headline1_tokens = nltk.word_tokenize(news_headlines[0])
# news_headline2_tokens = nltk.word_tokenize(news_headlines[1])

# news_headline3_tokens = nltk.word_tokenize(news_headlines[2])

# news_headline4_tokens = nltk.word_tokenize(news_headlines[3])

words = [word for word in news_headline1_tokens if len(word) > 1]
tensed = []
# words = ['gave', 'went', 'going', 'dating']
for word in words:
    c = WordNetLemmatizer().lemmatize(word, 'v')
    tensed.append(c)


poscleared = nltk.pos_tag(tensed)
# print(poscleared)

# grammar = r"""
#   NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
#   PP: {<IN><NP>}               # Chunk prepositions followed by NP
#   VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
#   CLAUSE: {<NP><VP>}           # Chunk NP, VP
#   """
# chunkParser = nltk.RegexpParser(grammar)
# chunker = RegexpParser("""
#                        NP: {<DT>?<JJ>*<NN>}    #To extract Noun Phrases
#                        P: {<IN>}               #To extract Prepositions
#                        V: {<V.*>}              #To extract Verbs
#                        PP: {<P> <NP>}          #To extract Prepostional Phrases
#                        VP: {<V> <NP|PP>*}      #To extarct Verb Phrases
#                        """)

# output = chunker.parse(poscleared)
# print(poscleared)
for word in poscleared:
    if (word[1] == "VBP" or word[1] == "VB"):  # Retrieve all adjectives.
        print(word)
# print(output)


# print("After Extracting\n", output)

stop_words = set(stopwords.words("english"))

filtered_sent = []
for w in tensed:
    if w not in stop_words:
        filtered_sent.append(w)
# print("Tokenized words:", news_headline1_tokens)
print("Filterd words:", filtered_sent)

fdist = FreqDist(filtered_sent)
# print(fdist)
print("\n Business Process Description: \n")
print(h1)
print("Frequency: \n")
for word, frequency in fdist.most_common(20):
    print(u'{};{}'.format(word, frequency))

# for words in [news_headline1_tokens, news_headline2_tokens, news_headline3_tokens, news_headline4_tokens]:
# for words in [news_headline1_tokens]:

#     print('First 7 tokens from news headlines: ', words[:7])


def transform(headlines):
    tokens = [w for s in headlines for w in s]
    print()
    # print('All Tokens:')
    # print(tokens)

    results = []
    label_enc = sklearn.preprocessing.LabelEncoder()
    onehot_enc = sklearn.preprocessing.OneHotEncoder()

    encoded_all_tokens = label_enc.fit_transform(list(set(tokens)))
    encoded_all_tokens = encoded_all_tokens.reshape(len(encoded_all_tokens), 1)

    onehot_enc.fit(encoded_all_tokens)

    for headline_tokens in headlines:
        print()
        # print('Original Input:', headline_tokens)

        encoded_words = label_enc.transform(headline_tokens)
        # print('Encoded by Label Encoder:', encoded_words)

        encoded_words = onehot_enc.transform(
            encoded_words.reshape(len(encoded_words), 1))
        # print('Encoded by OneHot Encoder:')
        # print(encoded_words)

        results.append(np.sum(encoded_words.toarray(), axis=0))

    return results


# transformed_results = transform([
#     news_headline1_tokens, news_headline2_tokens, news_headline3_tokens, news_headline4_tokens])
transformed_results = transform([
    news_headline1_tokens])

for j in range(len(news_headlines)):
    print('Original Description: %s' % news_headlines[j])

    for i, news_headline in enumerate(news_headlines):
        if i <= j:
            continue
        else:

            # print("i:", i)

            score = sklearn.metrics.pairwise.euclidean_distances(
                [transformed_results[i]], [transformed_results[0]])[0][0]
            print('-----')
            print('Score: %.2f, Comparing Sentence: %s' %
                  (score, news_headline))

# w1 = wordnet.synset('run.v.01')  # v here denotes the tag verb
# w2 = wordnet.synset('sprint.v.01')
# print(w1.wup_similarity(w2))
