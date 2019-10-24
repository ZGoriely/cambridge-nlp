from nltk.stem import PorterStemmer
import glob

ps = PorterStemmer()

def stemFile(pathA, pathB):
    fileA = open(pathA, 'r')
    fileB = open(pathB, 'w')

    data = list(map(lambda word: word.replace('\n',''), fileA.readlines()))
    for word in data:
        stem = ps.stem(word)
        fileB.writelines(stem+"\n")

    fileA.close()
    fileB.close()

def stemAllFiles():
    posFiles = glob.glob("./data/POS/*")
    for file in posFiles:
        newName = "./data/POSSTEM/"+file.split('_')[0].split('/')[-1]+".tag"
        stemFile(file,newName)
    negFiles = glob.glob("./data/NEG/*")
    for file in negFiles:
        newName = "./data/NEGSTEM/"+file.split('_')[0].split('/')[-1]+".tag"
        stemFile(file,newName)
stemAllFiles()    