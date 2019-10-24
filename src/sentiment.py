import glob
import bayes
import math
import svm
import scipy.stats

dataDirectoryPOS = "./data/POSSTEM/*"
dataDirectoryNEG = "./data/NEGSTEM/*"

# Split the dataset into n folds using round robin
def roundRobinSplit(n):

    posFiles = glob.glob(dataDirectoryPOS)
    negFiles = glob.glob(dataDirectoryNEG)
    list.sort(posFiles)
    list.sort(negFiles)

    folds = []
    for i in range(n):
        split = {"POS" : [], "NEG" : []}
        for j in range(math.ceil(len(posFiles)/n)):
            if (j*n+i >= len(posFiles)): break
            split["POS"].append(posFiles[j*n+i])
            split["NEG"].append(negFiles[j*n+i])
        folds.append(split)

    return folds

# Cross validate Bayes
def crossValidateBayes(folds):

    numFolds = len(folds)
    scores = [0.0]*numFolds

    # Select each fold for validation once only
    for i in range(numFolds):
        # Get test and training sets
        testSetPOS = folds[i]["POS"]
        testSetNEG = folds[i]["NEG"]
        trainingSetPOS = []
        trainingSetNEG = []
        for j in range(numFolds):
            if (i != j):
                trainingSetPOS.extend(folds[j]["POS"])
                trainingSetNEG.extend(folds[j]["NEG"])
        
        # Naive bayes on test set
        classProbabilities = {"POS" : 0.5, "NEG" : 0.5}
        logProbs = bayes.calculateSmoothedLogProbs(trainingSetPOS, trainingSetNEG)
        predictionsPOS = bayes.naiveBayes(testSetPOS, logProbs, classProbabilities)
        predictionsNEG = bayes.naiveBayes(testSetNEG, logProbs, classProbabilities)

        # Calculate accuracy
        correctPOS = sum(map(lambda x: 1 if (x == "POS") else 0, predictionsPOS.values()))
        correctNEG = sum(map(lambda x: 1 if (x == "NEG") else 0, predictionsNEG.values()))
        scores[i] = (correctPOS+correctNEG)/(len(testSetPOS)+len(testSetNEG))

    return sum(scores)/numFolds

# Cross validate SVM
def crossValidateSVM(folds):

    numFolds = len(folds)
    scores = [0.0]*numFolds

    # Select each fold for validation once only
    for i in range(numFolds):
        # Get test and training sets
        testSetPOS = folds[i]["POS"]
        testSetNEG = folds[i]["NEG"]
        trainingSetPOS = []
        trainingSetNEG = []
        for j in range(numFolds):
            if (i != j):
                trainingSetPOS.extend(folds[j]["POS"])
                trainingSetNEG.extend(folds[j]["NEG"])
        
        # SVM on test set
        model, wordMap = svm.train(trainingSetPOS, trainingSetNEG)
        predictionsPOS = svm.predict(model, testSetPOS, wordMap)
        predictionsNEG = svm.predict(model, testSetNEG, wordMap)

        # Calculate accuracy
        correctPOS = sum(map(lambda x: 1 if (x == "POS") else 0, predictionsPOS.values()))
        correctNEG = sum(map(lambda x: 1 if (x == "NEG") else 0, predictionsNEG.values()))
        scores[i] = (correctPOS+correctNEG)/(len(testSetPOS)+len(testSetNEG))

    return sum(scores)/numFolds

# Calculates the accuracy of a system
def calculateAccuracy(trueSentiments, predictions):
    correct = 0
    for file, sentiment in trueSentiments.items():
        if (predictions[file] == sentiment): correct+=1
    return correct/len(trueSentiments)

# Performs the sign test on two systems
def signTest(trueSentiments, predictionsA, predictionsB):

    # Tally the performance of both systems
    plus = 0
    minus = 0
    null = 0
    correctA = 0
    correctB = 0
    for file, sentiment in trueSentiments.items():
        if (predictionsA[file] == predictionsB[file] and predictionsB[file] == sentiment):
            # Systems same
            null+=1
            correctA+=1
            correctB+=1
        elif (predictionsA[file] == sentiment):
            # A outperforms B
            plus+=1             
            correctA+=1         
        elif (predictionsB[file] == sentiment):
            # B outperforms A
            minus+=1         
            correctB+=1             

    # Do sign test calculation
    k = math.ceil(null/2)+min(plus, minus)
    n = 2*math.ceil(null/2)+plus+minus
    p = scipy.stats.binom_test(k, n)

    return p

# Gives the accuracies for each system for each fold, the average accuracies
# and the p value of the two systems
def compareTwoSystems(folds, systemA, systemB, systemAName, systemBName):

    numFolds = len(folds)
    scoresA = []
    scoresB = []
    allTrueSentiments= {}
    allPredictionsA = {}
    allPredictionsB = {}

    for i in range(numFolds):
        # Get test and training sets
        testSetPOS = folds[i]["POS"]
        testSetNEG = folds[i]["NEG"]
        trainingSetPOS = []
        trainingSetNEG = []
        for j in range(numFolds):
            if (i != j):
                trainingSetPOS.extend(folds[j]["POS"])
                trainingSetNEG.extend(folds[j]["NEG"])

        # Generate true sentiments for test set
        trueSentiments = {}
        for file in testSetPOS: trueSentiments[file] = "POS"
        for file in testSetNEG: trueSentiments[file] = "NEG"

        # Get predictions from both systems
        predictionA = systemA(trainingSetPOS, trainingSetNEG, testSetPOS, testSetNEG)
        predictionB = systemB(trainingSetPOS, trainingSetNEG, testSetPOS, testSetNEG)
        
        # Add predictions for sign test
        allTrueSentiments.update(trueSentiments)
        allPredictionsA.update(predictionA)
        allPredictionsB.update(predictionB)

        # Get accuracies for this fold
        accA = calculateAccuracy(trueSentiments, predictionA)
        accB = calculateAccuracy(trueSentiments, predictionB)
        scoresA.append(accA)
        scoresB.append(accB)

        print("=========FOLD ", i,"========", sep="")
        print("Accuracy of ", systemAName, ": ", accA, sep="")
        print("Accuracy of ", systemBName, ": ", accB, sep="")
        print()

    p = signTest(allTrueSentiments, allPredictionsA, allPredictionsB)

    print("=========AVERAGES=========")
    print("Average", systemAName, "Accuracy:", sum(scoresA)/numFolds)
    print("Average", systemBName, "Accuracy:", sum(scoresB)/numFolds)
    print("p value:", p)

# ======================================================== #
# ------------------ Possible Experiments ---------------- #
# ======================================================== #

# Bayes with Frequency and Unigrams
def bayesFrequencyUnigrams(trainingSetPOS, trainingSetNEG, testSetPOS, testSetNEG):
    prediction = {}
    classProbabilities = {"POS" : 0.5, "NEG" : 0.5}
    logProbs = bayes.calculateSmoothedLogProbs(trainingSetPOS, trainingSetNEG)
    prediction = bayes.naiveBayes(testSetPOS, logProbs, classProbabilities)
    prediction.update(bayes.naiveBayes(testSetNEG, logProbs, classProbabilities))
    return prediction

def svmFrequencyUnigrams(trainingSetPOS, trainingSetNEG, testSetPOS, testSetNEG):
    prediction = {}
    model, wordMap = svm.train(trainingSetPOS, trainingSetNEG)
    prediction = svm.predict(model, testSetPOS, wordMap)
    prediction.update(svm.predict(model, testSetNEG, wordMap))
    return prediction

def bayesPresenceUnigrams(trainingSetPOS, trainingSetNEG, testSetPOS, testSetNEG):
    prediction = {}
    classProbabilities = {"POS" : 0.5, "NEG" : 0.5}
    logProbs = bayes.calculateSmoothedLogProbs(trainingSetPOS, trainingSetNEG,presence=True)
    prediction = bayes.naiveBayes(testSetPOS, logProbs, classProbabilities,presence=True)
    prediction.update(bayes.naiveBayes(testSetNEG, logProbs, classProbabilities,presence=True))
    return prediction

def svmPresenceUnigrams(trainingSetPOS, trainingSetNEG, testSetPOS, testSetNEG):
    prediction = {}
    model, wordMap = svm.train(trainingSetPOS, trainingSetNEG, presence=True)
    prediction = svm.predict(model, testSetPOS, wordMap, presence=True)
    prediction.update(svm.predict(model, testSetNEG, wordMap,presence=True))
    return prediction

def bayesPresenceBigrams(trainingSetPOS, trainingSetNEG, testSetPOS, testSetNEG):
    prediction = {}
    classProbabilities = {"POS" : 0.5, "NEG" : 0.5}
    logProbs = bayes.calculateSmoothedLogProbs(trainingSetPOS, trainingSetNEG, unigrams=False, bigrams=True,presence=True)
    prediction = bayes.naiveBayes(testSetPOS, logProbs, classProbabilities, unigrams=False, bigrams=True, presence=True)
    prediction.update(bayes.naiveBayes(testSetNEG, logProbs, classProbabilities, unigrams=False, bigrams=True, presence=True))
    return prediction

def svmPresenceBigrams(trainingSetPOS, trainingSetNEG, testSetPOS, testSetNEG):
    prediction = {}
    model, wordMap = svm.train(trainingSetPOS, trainingSetNEG, unigrams=False, bigrams=True, presence=True)
    prediction = svm.predict(model, testSetPOS, wordMap, unigrams=False, bigrams=True, presence=True)
    prediction.update(svm.predict(model, testSetNEG, wordMap, unigrams=False, bigrams=True, presence=True))
    return prediction

def bayesPresenceUnigramsBigrams(trainingSetPOS, trainingSetNEG, testSetPOS, testSetNEG):
    prediction = {}
    classProbabilities = {"POS" : 0.5, "NEG" : 0.5}
    logProbs = bayes.calculateSmoothedLogProbs(trainingSetPOS, trainingSetNEG, bigrams=True,presence=True)
    prediction = bayes.naiveBayes(testSetPOS, logProbs, classProbabilities, bigrams=True, presence=True)
    prediction.update(bayes.naiveBayes(testSetNEG, logProbs, classProbabilities, bigrams=True, presence=True))
    return prediction

def svmPresenceUnigramsBigrams(trainingSetPOS, trainingSetNEG, testSetPOS, testSetNEG):
    prediction = {}
    model, wordMap = svm.train(trainingSetPOS, trainingSetNEG, bigrams=True, presence=True)
    prediction = svm.predict(model, testSetPOS, wordMap, bigrams=True, presence=True)
    prediction.update(svm.predict(model, testSetNEG, wordMap, bigrams=True, presence=True))
    return prediction

#print(crossValidateBayes(roundRobinSplit(3)))

#compareTwoSystems(roundRobinSplit(10), bayesPresenceUnigrams, bayesPresenceBigrams, "Bayes P Unigrams", "Bayes P Bigrams")
compareTwoSystems(roundRobinSplit(10), svmFrequencyUnigrams, bayesFrequencyUnigrams, "bfu", "bpu")
