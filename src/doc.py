import gensim
from gensim import corpora
import glob
import pprint
from six import iteritems

posTrainFiles = "./data/aclImdb/train/pos/*"
negTrainFiles = "./data/aclImdb/train/neg/*"
posTestFiles = "./data/aclImdb/test/pos/*"
negTestFiles = "./data/aclImdb/test/neg/*"
usupTrainFiles = "./data/aclImdb/train/unsup/*"

modelFile = "./tmp/doc.model"

def read_corpus(dataSet, tokens_only=False):
    for i in range(len(dataSet)):
        documentFile = dataSet[i]
        with open(documentFile) as document:
            line = document.readline()
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = None

# trainFiles = glob.glob(posTrainFiles)
# trainFiles.extend(glob.glob(negTrainFiles))
# trainFiles.extend(glob.glob(posTestFiles))
# trainFiles.extend(glob.glob(negTestFiles))
# trainFiles.extend(glob.glob(usupTrainFiles))
# #trainFiles.sort()
# train_corpus = list(read_corpus(trainFiles))

def createDocModel(parameter):
    # Create the doc2vec model
    # Best so far, 87%: model = gensim.models.doc2vec.Doc2Vec(dm=0, vector_size=110, min_count=2, epochs=6, workers=8, hs=0, window=6)
    my_dm = 0
    my_vector_size = 120
    my_min_count = 2
    my_epochs = 10
    my_hs = 1
    my_window = 20
    my_negative = 10
    #my_dm = 0
    #my_vector_size = 111
    #my_min_count = 4
    #my_epochs = 16
    #my_hs = 1
    #my_window = 46
    #my_negative = 10

    print("dm:", my_dm, "vector size:", my_vector_size, "min count:", my_min_count, "epochs:", my_epochs, "hs:", my_hs, "window:", my_window, "negative", my_negative)

    model = gensim.models.doc2vec.Doc2Vec(seed=0, dm=my_dm,
        vector_size=my_vector_size, min_count=my_min_count, epochs=my_epochs, workers=1, hs=my_hs, window=my_window, negative=my_negative)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # Reduce memory usage
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    # Save the model
    model.save(modelFile)
    return model

# createDocModel()

#model = gensim.models.Doc2Vec.load(modelFile)

# vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])
# print(vector)

# ranks = []
# second_ranks = []
# for doc_id in range(len(train_corpus)):
#     inferred_vector = model.infer_vector(train_corpus[doc_id].words)
#     sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
#     rank = [docid for docid, sim in sims].index(doc_id)
#     ranks.append(rank)

#     second_ranks.append(sims[1])

# import collections

# counter = collections.Counter(ranks)
# print(counter)

# print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
# print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
# for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
#     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

# import random

# test_corpus = list(read_corpus(glob.glob(posTestFiles),tokens_only=True))

# # Pick a random document from the test corpus and infer a vector from the model
# doc_id = random.randint(0, len(test_corpus) - 1)
# inferred_vector = model.infer_vector(test_corpus[doc_id])
# sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# # Compare and print the most/median/least similar documents from the train corpus
# print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
# print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
# for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
#     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))