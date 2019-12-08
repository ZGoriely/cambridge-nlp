from sklearn.manifold import TSNE
import glob
import gensim
from svm import convertDoc2VecToSVMVec
from sklearn.decomposition import TruncatedSVD
import bayes
import pickle
import nltk
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import matplotlib.ticker as plticker
from nltk.corpus import wordnet
from itertools import product
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


def getPOS():
    words = "what should you do if your iron starts to drip water ?".split(" ")
    print(nltk.pos_tag(words))

def wordnettest():
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets("good"):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    print(set(synonyms))
    print(set(antonyms))

def mostSimilarToDrip(compareWord):

    ps = PorterStemmer()
    with open('./data/text1.txt') as text:
        data = gensim.utils.simple_preprocess(text.read())
        data = [ps.stem(word) for word in data]

    drip = wordnet.synsets('drip')

    allsyns1 = set(ss for word in data for ss in wordnet.synsets(word))
    similarities = [(wordnet.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, [wordnet.synset(compareWord)])]
    similarities = sorted(similarities, key=lambda x: x[0])
    for i in similarities:
        print(i)
    
def mostSimilarToDripWordVec(compareWord):

    ps = PorterStemmer()
    with open('./data/text1.txt') as text:
        data = gensim.utils.simple_preprocess(text.read())
        data = [ps.stem(word) for word in data]


    model = gensim.models.KeyedVectors.load_word2vec_format('./tmp/GoogleNews-vectors-negative300.bin', binary=True)

    seen = []

    similarities = []
    for word in data:
        if word in seen:
            continue
        seen.append(word)
        try:
            similarity = model.similarity(word, compareWord)
            similarities.append((similarity, word, compareWord))
        except:
            continue

    similarities = sorted(similarities, key=lambda x: x[0])
    for i in similarities:
        print(i)

    
def compareSentences():

    with open('./data/text1.txt') as text:
        data = gensim.utils.simple_preprocess(text.read())

    model = gensim.models.KeyedVectors.load_word2vec_format('./tmp/GoogleNews-vectors-negative300.bin', binary=True)



mostSimilarToDrip('quantity.n.01')
#mostSimilarToDripWordVec('quantity')
#getPOS()


