import svmlight
import bayes

def train(trainingPOS, trainingNEG, unigrams=True, bigrams=False, presence=False):

    # Create map from word to feature number
    unigramMap = {}
    bigramMap = {}
    createWordMap(unigramMap, bigramMap, trainingPOS, trainingNEG)
    wordMap = (unigramMap, bigramMap)

    # Create document vectors for each files and add to training_data
    training_data = []
    for file in trainingPOS:
        wordList = bayes.load_file(file)
        docVec = getDocumentVector(wordList, wordMap, unigrams, bigrams, presence)
        training_data.append((1, docVec))
    for file in trainingNEG:
        wordList = bayes.load_file(file)
        docVec = getDocumentVector(wordList, wordMap, unigrams, bigrams, presence)
        training_data.append((-1, docVec))

    # Generate the model from the training data
    model = svmlight.learn(training_data, type='classification')

    return model, wordMap

def predict(model, testSet, wordMap, unigrams=True, bigrams=False, presence=False):

    results = {}

    # Put test set into correct format
    test_data = []
    for file in testSet:
        wordList = bayes.load_file(file)
        docVec = getDocumentVector(wordList, wordMap, unigrams, bigrams, presence)
        test_data.append((0, docVec))

    # Get predictions from the model
    predictions = svmlight.classify(model, test_data)

    # Return predictions in the same format as the bayes
    for i in range(len(predictions)):
        results[testSet[i]] = "POS" if predictions[i] > 0 else "NEG"

    return results

def getDocumentVector(wordList, wordMap, unigrams=True, bigrams=False, presence=False):

    unigramMap, bigramMap = wordMap

    docVec = []
    wordCount = {}

    # Create the word vector as a dictionary
    if (unigrams):
        for unigram in wordList:
            if not unigram in unigramMap: continue # If don't recognise word
            fid = unigramMap[unigram]
            if presence: wordCount[fid] = 1
            else:        wordCount[fid] = 1 if not (fid in wordCount) else wordCount[fid] + 1
    if (bigrams):
        for bigram in zip(wordList[:-1], wordList[1:]):
            if not bigram in bigramMap: continue
            fid = bigramMap[bigram]
            if presence: wordCount[fid] = 1
            else:        wordCount[fid] = 1 if not (fid in wordCount) else wordCount[fid] + 1

    # Convert the word vector to a list of (wordID, count) pairs
    for wordID, count in wordCount.items():
        docVec.append((wordID, count))
    list.sort(docVec)
    return docVec

def createWordMap(unigramMap, bigramMap, posData, negData):
    # Loop through all data to create a map from words to feature IDs
    i = 1
    for file in posData:
        wordList = bayes.load_file(file)
        for unigram in wordList:
            # Create new entry if word hasn't been added yet
            if not (unigram in unigramMap):
                unigramMap[unigram] = i
                i+=1
        for bigram in zip(wordList[:-1], wordList[1:]):
            # Create new entry if word hasn't been added yet
            if not (bigram in bigramMap):
                bigramMap[bigram] = i
                i+=1
    for file in negData:
        wordList = bayes.load_file(file)
        for unigram in wordList:
            if not (unigram in unigramMap):
                unigramMap[unigram] = i
                i+=1
        for bigram in zip(wordList[:-1], wordList[1:]):
            # Create new entry if word hasn't been added yet
            if not (bigram in bigramMap):
                bigramMap[bigram] = i
                i+=1
