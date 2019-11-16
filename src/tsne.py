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

docFile = "./tmp/data.vecs"
tags = {
"CD" :  "black"
,"CC" :  "black"
,"DT" :  "black"
,"EX" :  "black"
,"FW" :  "slategrey"
,"IN" :  "gold"
,"JJR" :  "green"
,"JJS" : "green"
,"JJ" : "green"
,"LS" :  "black"
,"MD" :  "red"
,"NN" :  "blue"
,"NNS" : "blue"
,"NNP" : "purple"
,"NNPS" : "purple"
,"PDT" : "black"
,"POS" : "black"
,"PRP" : "darkorange"
,"PRP$" : "darkorange"
,"RB" :  "pink"
,"RBR" : "pink"
,"RBS" : "pink"
,"RP" :  "black"
,"TO" :  "black"
,"UH" :  "peru"
,"VB" :  "red"
,"VBD" : "red"
,"VBN" : "red"
,"VBG" : "red"
,"VBP" : "red"
,"VBZ" : "red"
,"WDT" : "black"
,"WP" :  "darkorange"
,"WP$" : "darkorange"
,"WRB" : "pink"}

def documents_to_vector_dataset():
    dataDirectoryPOS = "./data/POS/*"
    dataDirectoryNEG = "./data/NEG/*"

    posFiles = glob.glob(dataDirectoryPOS)
    negFiles = glob.glob(dataDirectoryNEG)
    posFiles.sort()
    negFiles.sort()

    docModel = gensim.models.Doc2Vec.load("./tmp/doc87.model")

    posVecs = []
    posTargets = []
    negVecs = []
    negTargets = []
    for file in posFiles:
        wordList = bayes.load_file(file)
        docVec = docModel.infer_vector(wordList)
        posVecs.append(docVec)
        posTargets.append(0)
    for file in negFiles:
        wordList = bayes.load_file(file)
        docVec = docModel.infer_vector(wordList)
        negVecs.append(docVec)
        negTargets.append(1)

    posFileNames = ["".join(file.split("_")[0].split("/")[2:4]) for file in posFiles]
    negFileNames = ["".join(file.split("_")[0].split("/")[2:4])for file in negFiles]
    
    dataset = {"posVecs" : posVecs, "negVecs" : negVecs, "posTargets" : posTargets,
    "negTargets" : negTargets, "posFiles" : posFiles, "negFiles" : negFiles,
    "posFileNames" : posFileNames, "negFileNames" : negFileNames}

    output = open(docFile, 'wb')
    pickle.dump(dataset, output)
    output.close()

def do_tsne():
    pkl_file = open(docFile, 'rb')
    dataset = pickle.load(pkl_file)
    pkl_file.close()   

    #X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(dataset["vecs"])
    #X_embedded = TSNE(n_components=2, perplexity=5, verbose=2).fit_transform(X_reduced)
    #X_tsne = TSNE(learning_rate=100).fit_transform(vecs)
    X_tsne = TSNE(learning_rate=100, n_components=2, perplexity=5, verbose=2).fit_transform(dataset["posVecs"][0:100] + dataset["negVecs"][0:100])

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=dataset["posTargets"][0:100] + dataset["negTargets"][0:100])

    plt.show()

def do_tsne2():
    pkl_file = open(docFile, 'rb')
    dataset = pickle.load(pkl_file)
    pkl_file.close()   

    #X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(dataset["vecs"])
    #X_embedded = TSNE(n_components=2, perplexity=5, verbose=2).fit_transform(X_reduced)
    #X_tsne = TSNE(learning_rate=100).fit_transform(vecs)

    n = 30

    X_tsne = TSNE(n_components=2, perplexity=5, verbose=1).fit_transform(dataset["posVecs"][:n] + dataset["negVecs"][:n])

    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=dataset["posTargets"][:n] + dataset["negTargets"][:n])

    for i, txt in enumerate(dataset["posFileNames"][:n] + dataset["negFileNames"][:n]):
        ax.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]))

    plt.show()

def tnse_one_doc():
    pkl_file = open(docFile, 'rb')
    dataset = pickle.load(pkl_file)
    pkl_file.close()   

    d = 2
    n = 100

    vec = dataset["posVecs"][d]
    wordList = bayes.load_file(dataset["posFiles"][d])

    docModel = gensim.models.Doc2Vec.load("./tmp/doc87.model")

    wordVecs = []
    wordTypes = []
    words = []
    for word in wordList:
        try:
            wordVec = docModel.wv.get_vector(word)
        except(KeyError):
            continue
        #wordVec = docModel.infer_vector([str(word)])
        tag = tags[nltk.pos_tag([word])[0][1]]
        if word not in words and tag in ["blue", "red", "pink", "green"]:
            wordTypes.append(tag)
            wordVecs.append(wordVec)
            words.append(word)

    X_tsne = TSNE(perplexity=5, verbose=2).fit_transform(wordVecs + [vec])

    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=wordTypes+["silver"])

    for i, txt in enumerate(words + [dataset["posFileNames"][d]]):
        ax.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]))
    #ax.annotate("DOCUMENT", (X_tsne[0, 0], X_tsne[0, 1]))
    plt.show()

def tnse_n_doc():
    pkl_file = open(docFile, 'rb')
    dataset = pickle.load(pkl_file)
    pkl_file.close()   

    n = 40
    docnum = 5

    vecs = dataset["posVecs"][:docnum] + dataset["negVecs"][:docnum]

    docModel = gensim.models.Doc2Vec.load("./tmp/doc87.model")

    wordVecs = []
    words = []
    wordTypes = []
    for i in range(docnum):
        wordList = bayes.load_file(dataset["posFiles"][i]) + bayes.load_file(dataset["negFiles"][i])
        for word in wordList:
            try:
                wordVec = docModel.wv.get_vector(word)
            except(KeyError):
                continue
            #wordVec = docModel.infer_vector([str(word)])
            tag = tags[nltk.pos_tag([word])[0][1]]
            if (word not in words and tag in ["pink", "red"]):
                words.append(word)
                wordTypes.append(tag)
                wordVecs.append(wordVec)

    X_tsne = TSNE(n_components=2, perplexity=5, verbose=2).fit_transform(vecs + wordVecs)

    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=["silver" for i in range(2*docnum)]+wordTypes)

    for i, txt in enumerate(dataset["posFileNames"][:docnum] + dataset["negFileNames"][:docnum] + words):
        ax.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]))

    plt.show()

def tnse_manual():
    pkl_file = open(docFile, 'rb')
    dataset = pickle.load(pkl_file)
    pkl_file.close()   

    n = 40
    docnum = 0

    wordList = "characters were very convincing and felt like you could understand their feelings very enjoyable movie".split(" ")

    docModel = gensim.models.Doc2Vec.load("./tmp/doc87.model")

    vec = docModel.infer_vector(wordList)

    wordVecs = []
    words = []
    wordTypes = []
    for word in wordList:
        wordVec = docModel.infer_vector([str(word)])
        tag = tags[nltk.pos_tag([word])[0][1]]
        if (word not in words):
            words.append(word)
            wordTypes.append(tag)
            wordVecs.append(wordVec)

    X_tsne = TSNE(n_components=2, perplexity=2, verbose=2).fit_transform([vec] + wordVecs)

    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=["silver"]+wordTypes)

    for i, txt in enumerate(["DOC"] + words):
        ax.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]))

    plt.show()

def tnse_manual2():
    pkl_file = open(docFile, 'rb')
    dataset = pickle.load(pkl_file)
    pkl_file.close()   

    n = 40
    docnum = 20

    #wordList = ["best", "worst", "good", "bad", "horrible", "wonderful", "awful", "great", "mediocre", "average", "director", "actor", "film", "woman", "man"]
    #wordList = "best worst good bad wonderful awful film movie director actor actress".split(" ")
    wordList = "woman man actor actress director film movie plot story".split(" ")
    randomWords = "useful and think to by am that now going are a make i you than with is like".split(" ")
    wordList.extend(randomWords)

    docModel = gensim.models.Doc2Vec.load("./tmp/doc87.model")

    vec = docModel.infer_vector(wordList)

    wordVecs = []
    words = []
    wordTypes = []
    for word in wordList:
        wordVec = docModel.infer_vector([str(word)])
        print(nltk.pos_tag([word]))
        tag = tags[nltk.pos_tag([word])[0][1]]
        if (word not in words):
            words.append(word)
            wordTypes.append(tag)
            wordVecs.append(wordVec)

    X_tsne = TSNE(n_components=2, perplexity=3, verbose=2).fit_transform(wordVecs)

    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=wordTypes)

    for i, txt in enumerate(words):
        ax.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]))

    plt.show()

def tnse_most_frequent():

    n = 20
    docnum = 6

    pkl_file = open(docFile, 'rb')
    dataset = pickle.load(pkl_file)
    pkl_file.close()   

    allWords = {}
    for file in dataset["posFiles"]:
        wordList = bayes.load_file(file)
        for word in wordList:
            if word not in allWords:
                allWords[word] = 0
            allWords[word] += 1  
    for file in dataset["negFiles"]:
        wordList = bayes.load_file(file)
        for word in wordList:
            if word not in allWords:
                allWords[word] = 0
            allWords[word] += 1     


    wordList = sorted(allWords, key=allWords.__getitem__)
    wordListAdjective = list(filter(lambda word: tags[nltk.pos_tag([word])[0][1]] == "green", wordList))[-n:]
    wordListNoun = list(filter(lambda word: tags[nltk.pos_tag([word])[0][1]] == "blue", wordList))[-n:]
    wordListProperNoun = list(filter(lambda word: tags[nltk.pos_tag([word])[0][1]] == "purple", wordList))[-n:]
    wordListVerb = list(filter(lambda word: tags[nltk.pos_tag([word])[0][1]] == "red", wordList))[-n:]
    wordListAdverb = list(filter(lambda word: tags[nltk.pos_tag([word])[0][1]] == "pink", wordList))[-n:]
    wordListPersonalPronoun = list(filter(lambda word: tags[nltk.pos_tag([word])[0][1]] == "darkorange", wordList))[-n:]
    #wordList = wordListAdjective + wordListNoun
    wordList = wordListNoun + wordListAdjective
    print(wordList)

    docModel = gensim.models.Doc2Vec.load("./tmp/doc87.model")

    vec = docModel.infer_vector(wordList)

    wordVecs = []
    words = []
    wordTypes = []
    for word in wordList:
        try:
            wordVec = docModel.wv.get_vector(word)
        except(KeyError):
            continue
        #wordVec = docModel.infer_vector([str(word)])
        print(nltk.pos_tag([word]))
        tag = tags[nltk.pos_tag([word])[0][1]]
        if (word not in words):
            words.append(word)
            wordTypes.append(tag)
            wordVecs.append(wordVec)

    vecs = dataset["posVecs"][:docnum] + dataset["negVecs"][:docnum]

    X_tsne = TSNE(n_components=2, perplexity=5, verbose=2).fit_transform(wordVecs + vecs)

    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=wordTypes + ["grey"] * docnum + ["black"] * docnum)

    for i, txt in enumerate(words + dataset["posFileNames"][:docnum] + dataset["negFileNames"][:docnum]):
        ax.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]))

    plt.show()

def tnse_most_similar():

    pkl_file = open(docFile, 'rb')
    dataset = pickle.load(pkl_file)
    pkl_file.close()   

    docModel = gensim.models.Doc2Vec.load("./tmp/doc87.model")

    parentWords = ["good", "bad", "woman", "man"]
    childWords = []
    wordTypes = [0] * len(parentWords)
    for i, word in enumerate(parentWords):
        childWordList = docModel.wv.most_similar(word, topn=5)
        print("Most similar words to", word, "are: ")
        print(childWordList)
        childWords.extend([w for (w, _) in childWordList])
        wordTypes.extend([i+1] * len(childWordList))

    wordVecs = []
    words = []
    for word in parentWords+childWords:
        try:
            wordVec = docModel.wv.get_vector(word)
        except(KeyError):
            continue
        #wordVec = docModel.infer_vector([str(word)])
        tag = tags[nltk.pos_tag([word])[0][1]]
        if (word not in words):
            words.append(word)
            wordVecs.append(wordVec)

    X_tsne = TSNE(n_components=2, perplexity=5, verbose=2).fit_transform(wordVecs)

    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=wordTypes)

    for i, txt in enumerate(words):
        ax.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]))

    plt.show()

def tnse_composition(): 

    docModel = gensim.models.Doc2Vec.load("./tmp/doc87.model")

    randomWords = "useful and think to by am that now going are a make i you than with is like".split(" ")[:6]
    #pos = ["like a lot", "even better", "much better", "so nice", "so good", "really good", "very nice", "better",
    #        "very good", "love so much", "best", "incredibly good", "love", "amazing", "amazingly good", "fantastic",
    #        "terrific", "nice", "wonderful", "good", "interesting"]
    #negPos = ["not bad"]
    #neg = ["not interesting", "not good", "not good at all", "bad", "not useful at all", "not great", "not interesting at all",
    #        "dislike", "worst", "not useful", "not nice", "amazingly bad", "very bad", "hardly useful", "much worse", "so bad", "worse"]
    adj = ["good", "bad", "fantastic", "awful", "nice", "impressive"]
    not_adj = ["not " + a for a in adj]
    so_adj = ["so " + a for a in adj]
    very_adj = ["very " + a for a in adj]

    #phrases = randomWords + pos + negPos + neg
    phrases = randomWords + adj + not_adj + so_adj + very_adj
    allPhrases = [s.split(" ") for s in phrases]

    #wordTypes = [0] * len(randomWords) + [1] * len(pos) + [2] * len(negPos) + [3] * len(neg)
    wordTypes = [0] * len(randomWords) + [1] * len(adj) + [2] * len(not_adj) + [3] * len(so_adj) + [4] * len(very_adj)
    wordVecs = []
    for sentence in allPhrases:
        wordVec = docModel.infer_vector(sentence)
        wordVecs.append(wordVec)

    X_tsne = TSNE(n_components=2, perplexity=5, verbose=5).fit_transform(wordVecs)

    plt.rcParams.update({'font.size': 6})
    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=wordTypes)
    #i_good = phrases.index("good")
    #i_not_good = phrases.index("not good")
    #i_so_good = phrases.index("so good")
    #i_incredibly_good = phrases.index("incredibly good")
    kargs = {"head_width" : 3, "length_includes_head" : True, "color" : "black"}
    #ax.arrow(X_tsne[i_good, 0], X_tsne[i_good, 1], X_tsne[i_not_good, 0]-X_tsne[i_good, 0], X_tsne[i_not_good, 1]-X_tsne[i_good, 1], **kargs)
    #ax.arrow(X_tsne[i_good, 0], X_tsne[i_good, 1], X_tsne[i_so_good, 0]-X_tsne[i_good, 0], X_tsne[i_so_good, 1]-X_tsne[i_good, 1], **kargs)

    for i, txt in enumerate(phrases):
        ax.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]))

    plt.show()

def heatmap():
    docModel = gensim.models.Doc2Vec.load("./tmp/doc89.model")
    #vec1 = np.array([docModel.wv.get_vector("good")])
    #vec2 = np.array([docModel.infer_vector(["very", "good"])])
    sentences = ["good", "bad", "the movie was good", "the movie was not good","the movie was bad", "the movie was not bad"]
    sentenceArrays = [sentence.split(" ") for sentence in sentences]

    sentenceVectors = []
    for s in sentenceArrays:
        vec = docModel.infer_vector(s)
        sentenceVectors.append(np.array([vec]))

    pc_kwargs = {'rasterized': True, 'cmap': 'coolwarm'}
    fig, axs = plt.subplots(len(sentences), 1, figsize=(4, 4), constrained_layout=True)
    for i, ax in enumerate(axs):
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_title(sentences[i])
        im = ax.pcolormesh(sentenceVectors[i], **pc_kwargs)
        ax.xaxis.set_ticks_position('none') 
    #plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    fig.colorbar(im, ax=axs, shrink=0.6)

    plt.show()

def heatmap_sentence():
    docModel = gensim.models.Doc2Vec.load("./tmp/doc89.model")
    #vec1 = np.array([docModel.wv.get_vector("good")])
    #vec2 = np.array([docModel.infer_vector(["very", "good"])])
    sentence = "this has to be one of the greatest movies of all time"
    #sentence = "i thought that the movie was very bad"
    #sentence = "this was the worst film i have ever seen"
    #sentence = "i hate the movie though the plot is interesting"
    sentenceArray = sentence.split(" ")[::-1]

    wordVectors = []
    for word in sentenceArray:
        vec = docModel.infer_vector([word])
        wordVectors.append(np.array(vec))

    varianceVectors = []
    ns = len(wordVectors)
    for i in range(ns):
        wordVec = wordVectors[i]
        varyVec = np.array(wordVec)
        for j in range(len(wordVec)):
            varyVec[j] = wordVec[j]
            for i2 in range(ns):
                if not i2 == i:
                    varyVec[j] -= wordVectors[i2][j]/ns
            varyVec[j] = abs(varyVec[j]) ** 2
        varianceVectors.append(varyVec)

    print(wordVectors[0][0:10])
    print(varianceVectors[0][0:10])

    pc_kwargs = {'rasterized': True, 'cmap': 'Blues'}
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels(sentenceArray)
    loc = plticker.LinearLocator(len(wordVectors)+1)
    #loc = plticker.MultipleLocator(1) # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    dx = 0/72.; dy = 30/72. 
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)  
    for tick in ax.yaxis.get_major_ticks():
        print(tick)
    for label in ax.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    ax.set_title(sentence)
    im = ax.pcolormesh(varianceVectors, **pc_kwargs)
    ax.xaxis.set_ticks_position('none') 
    fig.colorbar(im, ax=ax, shrink=0.6)

    plt.show()

#documents_to_vector_dataset()
#do_tsne2()
#tnse_one_doc()
#tnse_n_doc()
tnse_manual()
#tnse_most_frequent()
#tnse_most_similar()

#heatmap_sentence()
#tnse_composition()

#docModel = gensim.models.Doc2Vec.load("./tmp/doc87.model")
#print(docModel.wv.distance("bad","bad"))
#print(docModel.wv.distance("bad","terrible"))
#print(docModel.wv.distance("bad","awful"))
#print(docModel.wv.distance("bad","good"))
