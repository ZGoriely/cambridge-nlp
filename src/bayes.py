import math
import itertools

def calculateSmoothedLogProbs(trainingPOS, trainingNEG, unigrams=True, bigrams=False, presence=False):

    # STEP 1
    # Create dictionary, count all words in document.
    # To smooth, start each value at 1

    totalUnigrams = {}
    totalBigrams = {}
    addToDictionary(totalUnigrams, totalBigrams, "POS", trainingPOS)
    addToDictionary(totalUnigrams, totalBigrams, "NEG", trainingNEG)

    # STEP 2
    # For each sentiment, count the total number of words from all documents
    # of that sentiment.

    sentimentTotals = {"POS" : 0, "NEG" : 0}
    if (unigrams):
        for unigramCount in totalUnigrams.values():
            for sentiment in sentimentTotals.keys():
                sentimentTotals[sentiment]+=unigramCount[sentiment]
    if (bigrams):
        for bigramCount in totalBigrams.values():
            for sentiment in sentimentTotals.keys():
                sentimentTotals[sentiment]+=bigramCount[sentiment]

    # STEP 3
    # Calculate the logarithm of the conditional probabilities for each word
    wordProbabilities = {}
    if (unigrams):
        for unigram, unigramCount in totalUnigrams.items():
            wordProb = {}
            for sentiment in sentimentTotals.keys():
                wordProb[sentiment] = math.log(unigramCount[sentiment]/(sentimentTotals[sentiment]+len(totalUnigrams)))
            wordProbabilities[unigram] = wordProb
    if (bigrams):
        for bigram, bigramCount in totalBigrams.items():
            wordProb = {}
            for sentiment in sentimentTotals.keys():
                wordProb[sentiment] = math.log(bigramCount[sentiment]/(sentimentTotals[sentiment]+len(totalBigrams)))
            wordProbabilities[bigram] = wordProb
    return wordProbabilities

def naiveBayes(testSet, tokenLogProbs, classProbabilities, unigrams=True, bigrams=False, presence=False):

    results = {}

    # Loop through each document to predict sentiment
    for file in testSet:
        wordList = load_file(file)

        # For each word, add the log probabilities to the respective sums
        posProbSum = 0
        negProbSum = 0
        unigramsSeen = []
        bigramsSeen = []
        if (unigrams):
            for unigram in wordList:
                if presence:
                    if unigram in unigramsSeen: continue
                    unigramsSeen.append(unigram)
                if unigram in tokenLogProbs:
                    posProbSum += tokenLogProbs[unigram]["POS"]
                    negProbSum += tokenLogProbs[unigram]["NEG"]
        if (bigrams):
            for bigram in zip(wordList[:-1], wordList[1:]):
                if presence:
                    if bigram in bigramsSeen: continue
                    bigramsSeen.append(bigram)
                if bigram in tokenLogProbs:
                    posProbSum += tokenLogProbs[bigram]["POS"]
                    negProbSum += tokenLogProbs[bigram]["NEG"]

        # Calculate the probabilities that the document is negative or positive,
        # then return the class of the larger probability
        posProb = math.log(classProbabilities["POS"]) + posProbSum
        negProb = math.log(classProbabilities["NEG"]) + negProbSum
        results[file] = "POS" if (posProb > negProb) else "NEG"

    return results

def load_file(path):
    with open(path,'r') as f:
        data = list(map(lambda word: word.replace('\n',''), f.readlines()))
    return data

def addToDictionary(dictUni, dictBi, sentiment, trainingData):
    for file in trainingData:

        wordList = load_file(file)

        # Get unigrams
        for unigram in wordList:
            # Create new entry if word hasn't been added yet
            if not (unigram in dictUni):
                wordCount = {"POS" : 1, "NEG" : 1}
                dictUni[unigram] = wordCount
            # Increment count of the word
            dictUni[unigram][sentiment] += 1

        # Get bigrams
        for bigram in zip(wordList[:-1], wordList[1:]):
            if not (bigram in dictBi):
                wordCount = {"POS" : 1, "NEG" : 1}
                dictBi[bigram] = wordCount
            dictBi[bigram][sentiment] += 1