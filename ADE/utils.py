# -*- coding: utf-8 -*-
import random
import gensim
import gzip
import numpy as np
import ast
import copy
import sys
from sklearn.model_selection  import train_test_split
from prettytable import PrettyTable
import re
import tensorflow as tf

"""Generic set of classes and methods"""


def strToLst(string):
    return ast.literal_eval(string)


class HeadData:
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def split(self, fraction):

        data_train, data_test, idx_train, idx_test = train_test_split(self.data, self.indices, test_size=fraction)
        train = HeadData(data_train, idx_train)
        test = HeadData(data_test, idx_test)
        return train, test


###run one time to obtain the characters
def getCharsFromDocuments(documents):
    chars = []
    for doc in documents:
        for tokens in doc.tokens:
            for char in tokens:
                # print (token)
                chars.append(char)
    chars = list(set(chars))
    chars.sort()
    return chars


###run one time to obtain the ner labels
def getEntitiesFromDocuments(documents):
    BIOtags = ['O', 'B-E', 'I-E'] ## 'E' denotes 'effect', 'D' denotes 'Drug'
    
    BIOtags = list(set(BIOtags))
    BIOtags.sort()
    
    return BIOtags


###run one time to obtain the relations
def getRelationsFromDocuments(documents):
    relations = ['Ade'] ## ade relation
    return relations


def getRelationNersFromDocuments(documents):
    bio_relation_ners = ['N', 'Ade__B-D', 'Ade__I-D']

    return bio_relation_ners


def getPosAndBio1s(text, tokens, ades):
    BIO1s = ['O'] * len(tokens)
    effect = {}
    drug = {}
    for ade in ades:
        e = ade[0]
        d = ade[1]
        eid = text.find(e)
        did = text.find(d)
        if eid == -1 or did == -1:
            continue
        elen = len(e.split(' '))
        dlen = len(d.split(' '))

        estart = len(text[:eid].strip().split(' '))
        dstart = len(text[:did].strip().split(' '))

        if e not in effect:
            effect[e] = (estart, estart+elen-1)
        if d not in drug:
            drug[d] = (dstart, dstart+dlen-1)
        
        for j in range(elen):
            if j == 0:
                BIO1s[estart + j] = 'B-E'
            else:
                BIO1s[estart + j] = 'I-E'

    return BIO1s, effect, drug


def getBIO2sAndIds(start, end, tokens, tags):
    BIOs = ['N'] * len(tokens)
    BIO_Ids = []
    for j in range(start, end + 1):
        if j == start:
            BIOs[j] = 'Ade__B-D'
        else:
            BIOs[j] = 'Ade__I-D'

    for bio in BIOs:
        id = tags.index(bio)
        BIO_Ids.append(id)

    return BIOs, BIO_Ids


def tokenToCharIds(token, characters):
    charIds = []
    for char in token:
        charIds.append(characters.index(char))
    return charIds


def labelsListToIds(listofLabels, setofLabels):
    labelIds = []
    for label in listofLabels:
        labelIds.append(setofLabels.index(label))

    return labelIds


def getLabelId(label, setofLabels):
    return setofLabels.index(label)


def strToBool(str):
    if str.lower() in ['true', '1']:
        return True
    return False


def getTokensFromText(text):
    tokens = []
    r ='([ !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~])' ## Punctuation
    segs = re.split(r, text)
    for seg in segs:
        if seg == ' ' or seg == '':
            continue
        tokens.append(seg)
    return tokens


def getEmbeddingId(word, embeddingsList):
    # modified method from http://cistern.cis.lmu.de/globalNormalization/globalNormalization_all.zip
    if word != "<empty>":
        if not word in embeddingsList:
            if re.search(r'^\d+$', word):
                word = "0"
            if word.islower():
                word = word.title()
            else:
                word = word.lower()
        if not word in embeddingsList:
            word = "<unk>"
        curIndex = embeddingsList[word]
        return curIndex


def readWordvectorsNumpy(documents, wordvectorfile, isBinary=False):

    # modified method from http://cistern.cis.lmu.de/globalNormalization/globalNormalization_all.zip
    wordvectors = []
    words = []
    doc_tokens = []

    for doc in documents:
        for tok in doc.tokens:
            if tok not in doc_tokens:
                doc_tokens.append(tok)

    model = gensim.models.KeyedVectors.load_word2vec_format(wordvectorfile, binary=isBinary,unicode_errors='ignore')

    vectorsize = model.vector_size

    random.seed(123456)
    randomVec = [random.uniform(-np.sqrt(1. / len(doc_tokens)), np.sqrt(1. / len(doc_tokens))) for i in range(vectorsize)]
    zeroVec = [0 for i in range(vectorsize)]

    wordvectors.append(zeroVec)
    words.append("<empty>")
    wordvectors.append(randomVec)
    words.append("<unk>")
    
    for key in doc_tokens:
        if key in model.wv:
            wordvectors.append(model.wv[key])
            words.append(key)

    wordvectorsNumpy = np.array(wordvectors)

    indices = {}
    for key in words:
        indices[key] = len(indices)

    return wordvectorsNumpy, vectorsize, words, indices


def readIndices(wordvectorfile, isBinary=False):
    # modified method from http://cistern.cis.lmu.de/globalNormalization/globalNormalization_all.zip
    indices = {}
    curIndex = 0
    indices["<empty>"] = curIndex
    curIndex += 1
    indices["<unk>"] = curIndex
    curIndex += 1

    model = gensim.models.KeyedVectors.load_word2vec_format(wordvectorfile, binary=isBinary,unicode_errors='ignore')

    count = 0
    # c=0
    for key in list(model.vocab.keys()):
        indices[key] = curIndex
        curIndex += 1

    return indices



def printParameters(config):

    t = PrettyTable(['Params', 'Value'])

    #dataset
    t.add_row(['Config', config.config_fname])
    t.add_row(['Embeddings', config.filename_embeddings])
    t.add_row(['Embeddings size ', config.representationsize])
    t.add_row(['Train', config.filename_train])
    t.add_row(['Dev', config.filename_dev])
    t.add_row(['Test', config.filename_test])

    #training
    t.add_row(['Epochs ', config.nepochs])
    t.add_row(['Optimizer ', config.optimizer])
    t.add_row(['Activation ', config.activation])
    t.add_row(['Learning rate ', config.learning_rate])
    t.add_row(['Gradient clipping ', config.gradientClipping])
    t.add_row(['Patience ', config.nepoch_no_imprv])
    t.add_row(['Use dropout', config.use_dropout])
    t.add_row(['Ner1 loss ', config.ner1_loss])
    t.add_row(['Ner2 loss ', config.ner2_loss])
    t.add_row(['Ner classes ', config.ner_classes])
    t.add_row(['Use char embeddings ', config.use_chars])
    t.add_row(['Use adversarial',config.use_adversarial])

    # hyperparameters
    t.add_row(['Dropout embedding ', config.dropout_embedding])
    t.add_row(['Dropout lstm ', config.dropout_lstm])
    t.add_row(['Dropout lstm output ', config.dropout_lstm_output])
    t.add_row(['Dropout fcl ner ', config.dropout_fcl_ner])
    t.add_row(['Dropout fcl rel ', config.dropout_fcl_rel])
    t.add_row(['Hidden lstm size ', config.hidden_size_lstm])
    t.add_row(['LSTM layers ', config.num_lstm_layers])
    t.add_row(['Attention heads ', config.num_heads])
    t.add_row(['Hidden nn size ', config.hidden_size_n1])
    t.add_row(['Char embeddings size ', config.char_embeddings_size])
    t.add_row(['Hidden size char ', config.hidden_size_char])
    t.add_row(['Label embeddings size ', config.label_embeddings_size])
    t.add_row(['Attention size ', config.attention_size])
    t.add_row(['Alpha ', config.alpha])
    t.add_row(['Root node ', config.root_node])

    #evaluation
    t.add_row(['Evaluation method ', config.evaluation_method])


    print(t)

def getDict(lst):
    return {k: v for v, k in enumerate(lst)}

def generator(data, m, config, train=False):
    # generate the data
    embeddingIds = m['embeddingIds']
    # isTrain=m['isTrain']

    # scoringMatrixGold = m['scoringMatrixGold']
    entity1_tags = m['entity1_tags'] # either the BIO tags or the EC tags - depends on the NER target values
    entity1_tags_ids = m['entity1_tags_ids']

    entity2_tags = m['entity2_tags'] # either the BIO tags or the EC tags - depends on the NER target values
    entity2_tags_ids = m['entity2_tags_ids']

    tokens = m['tokens']
    tokenIds = m['tokenIds']
    charIds = m['charIds']
    tokensLens = m['tokensLens']

    seqlen = m['seqlen']
    doc_ids=m['doc_ids']

    k1 = m['k1']
    k2 = m['k2']


    dropout_embedding_keep = m['dropout_embedding']
    dropout_lstm_keep = m['dropout_lstm']
    dropout_lstm_output_keep = m['dropout_lstm_output']
    dropout_fcl_ner_keep = m['dropout_fcl_ner']

    dropout_embedding_prob = 1
    dropout_lstm_prob = 1
    dropout_lstm_output_prob = 1
    dropout_fcl_ner_prob = 1
    dropout_fcl_rel_prob = 1

    if config.use_dropout == True and train==True:

        dropout_embedding_prob = config.dropout_embedding
        dropout_lstm_prob = config.dropout_lstm
        dropout_lstm_output_prob = config.dropout_lstm_output
        dropout_fcl_ner_prob = config.dropout_fcl_ner
        dropout_fcl_rel_prob = config.dropout_fcl_rel

    data_copy = copy.deepcopy(data)
    # train_ind=np.arange(len(train.data))
    if config.shuffle == True:
        shuffled_data, _, shuffled_data_idx, _ = train_test_split(data_copy.data, data_copy.indices, test_size=0, random_state=42)
        # shuffled_data, _, shuffled_data_idx, _ = train_test_split(data_copy.data, data_copy.indices, test_size=0,random_state=42)

        data_copy = HeadData(shuffled_data, shuffled_data_idx)
        # print ("shuffle:"+ str(shuffle) )
        # print(data_copy.indices)
    else:

        data_copy = HeadData(data_copy.data, data_copy.indices)
        # data_copy = HeadData(data_copy.data, data_copy.indices)

        # print("shuffle:" + str(shuffle))
        # print(data_copy.indices)

    # batchsize=16 # number of documents per batch
    batches_embeddingIds = []  # e.g., 131 batches
    batches_charIds = []  # e.g., 131 batches
    batches_tokens = []

    batches_entity1_tags = []
    batches_entity1_tags_ids = []
    batches_k1s = []
    batches_k2s = []
    batches_entity2_tags = []
    batches_entity2_tags_ids = []
    batches_tokenIds = []
    batches_doc_ids = []

    docs_batch_embeddingIds = []  # e.g., 587 max doc length - complete with -1 when the size of the doc is smaller
    docs_batch_charIds = []  # e.g., 587 max doc length - complete with -1 when the size of the doc is smaller

    docs_batch_entity1_tags=[] 
    docs_batch_entity1_tags_ids = []

    docs_batch_k1s = []
    docs_batch_k2s = []

    docs_batch_entity2_tags=[] 
    docs_batch_entity2_tags_ids = []

    docs_batch_tokens = []
    docs_batch_tokenIds = []
    docs_batch_doc_ids = []

    maxDocLenList = []
    maxSentenceLen = -1

    maxWordLenList = []
    maxWordLen = -1

    wordLenList = []
    wordLens = []

    lenBatchesDoc = []
    lenEmbeddingssDoc = []

    lenBatchesChars = []
    lenCharsDoc = []

    sumLen = 0
    for docIdx in range(len(data_copy.data)):
        doc = data_copy.data[docIdx]
        # print (doc)
        if docIdx % config.batchsize == 0 and docIdx > 0:
            # print (docIdx)
            # print ("new batch")
            batches_embeddingIds.append(docs_batch_embeddingIds)
            batches_charIds.append(docs_batch_charIds)

            batches_entity1_tags.append(docs_batch_entity1_tags)
            batches_entity1_tags_ids.append(docs_batch_entity1_tags_ids)

            batches_k1s.append(docs_batch_k1s)
            batches_k2s.append(docs_batch_k2s)

            batches_entity2_tags.append(docs_batch_entity2_tags)
            batches_entity2_tags_ids.append(docs_batch_entity2_tags_ids)

            batches_tokens.append(docs_batch_tokens)

            batches_tokenIds.append(docs_batch_tokenIds)
            batches_doc_ids.append(docs_batch_doc_ids)

            docs_batch_embeddingIds = []  # e.g., 587 max doc length - complete with -1 when the size of the doc is smaller
            docs_batch_charIds = []  # e.g., 587 max doc length - complete with -1 when the size of the doc is smaller

            docs_batch_tokens = []

            docs_batch_entity1_tags = []
            docs_batch_entity1_tags_ids = []

            docs_batch_k1s = []
            docs_batch_k2s = []

            docs_batch_entity2_tags = []
            docs_batch_entity2_tags_ids = []

            docs_batch_tokenIds = []
            docs_batch_doc_ids = []

            maxDocLenList.append(maxSentenceLen)
            maxSentenceLen = -1

            maxWordLenList.append(maxWordLen)
            maxWordLen = -1

            wordLenList.append(wordLens)

        if len(doc.token_ids) > maxSentenceLen:
            maxSentenceLen = len(doc.token_ids)

        longest_token_list=max(doc.char_ids, key=len)
        if len(longest_token_list) > maxWordLen:
            maxWordLen = len(longest_token_list)

        wordLens=[len(token) for token in doc.char_ids]

        sumLen += len(doc.token_ids)
        docs_batch_embeddingIds.append(doc.token_ids)
        docs_batch_charIds.append(doc.char_ids)

        
        docs_batch_entity1_tags.append(doc.BIO1s)##to do
        docs_batch_entity1_tags_ids.append(doc.BIO1_ids)

        ade = random.choice(doc.ades)
        epos = doc.effect[ade[0]]
        dpos = doc.drug[ade[1]]

        BIO2s, BIO2_ids = getBIO2sAndIds(dpos[0], dpos[1], doc.tokens, config.dataset_set_bio_relation_ners) 

        docs_batch_entity2_tags.append(BIO2s)##to do
        docs_batch_entity2_tags_ids.append(BIO2_ids)

        docs_batch_tokens.append(doc.tokens)
        
        docs_batch_k1s.append(epos[0])
        docs_batch_k2s.append(epos[1])

        docs_batch_tokenIds.append(doc.token_ids)
        docs_batch_doc_ids.append(doc.docId)

        if docIdx == len(data_copy.data) - 1:  ## if there are no documents left - append the batch - usually it is shorter batch
            batches_embeddingIds.append(docs_batch_embeddingIds)
            batches_charIds.append(docs_batch_charIds)

            batches_entity1_tags.append(docs_batch_entity1_tags)
            batches_entity1_tags_ids.append(docs_batch_entity1_tags_ids)
            batches_entity2_tags.append(docs_batch_entity2_tags)
            batches_entity2_tags_ids.append(docs_batch_entity2_tags_ids)

            batches_k1s.append(docs_batch_k1s)
            batches_k2s.append(docs_batch_k2s)

            batches_tokens.append(docs_batch_tokens)

            batches_tokenIds.append(docs_batch_tokenIds)
            batches_doc_ids.append(docs_batch_doc_ids)
            maxDocLenList.append(maxSentenceLen)
            maxWordLenList.append(maxWordLen)
            wordLenList.append(wordLens)
            # maxDocLen.append(maxWordLen)

    # print(len(batches_embeddingIds))
    for bIdx in range(len(batches_embeddingIds)):

        batch_embeddingIds = batches_embeddingIds[bIdx]
        batch_charIds = batches_charIds[bIdx]

        batch_entity1_tags = batches_entity1_tags[bIdx]
        batch_entity2_tags = batches_entity2_tags[bIdx]
        batch_tokens = batches_tokens[bIdx]

        batch_tokenIds = batches_tokenIds[bIdx]

        for dIdx in range(len(batch_embeddingIds)):
            embeddingId_doc = batch_embeddingIds[dIdx]
            charIds_doc = batch_charIds[dIdx]

            entity1_doc=batch_entity1_tags[dIdx]
            entity2_doc=batch_entity2_tags[dIdx]

            token_doc = batch_tokens[dIdx]
            token_id_doc = batch_tokenIds[dIdx]

            lenEmbeddingssDoc.append(len(embeddingId_doc))
            tokensLen=[len(token) for token in charIds_doc]
            lenCharsDoc.append(tokensLen)


            for tokenIdx in range(len(tokensLen)):
                tokenLen=tokensLen[tokenIdx]

                if tokenLen<maxWordLenList[bIdx]:

                    for i in np.arange(maxWordLenList[bIdx]-tokenLen):
                        #print (charIds_doc)
                        charIds_doc[tokenIdx].append(0)


            if len(embeddingId_doc) < maxDocLenList[bIdx]:
                # print  (maxWordLen-len(word_doc))
                # print ('here')
                for i in np.arange(maxDocLenList[bIdx] - len(embeddingId_doc)):
                    # pass
                    embeddingId_doc.append(0)
                    charIds_doc.append([])

                    token_doc.append("ZERO")

                    entity1_doc.append("ZERO")
                    entity2_doc.append("ZERO")
                    token_id_doc.append(maxDocLenList[bIdx] - 1)

        lenBatchesDoc.append(lenEmbeddingssDoc)

        lenBatchesChars.append(lenCharsDoc)

        lenEmbeddingssDoc = []
        lenCharsDoc=[]

    # return batches_words,batches_heads
    # print(len(batches_embeddingIds))
    for bIdx in range(len(batches_embeddingIds)):  # 131
        # print (bIdx)
        batch_embeddingIds = np.asarray(batches_embeddingIds[bIdx])  ## (batchsize, 36)
        batch_charIds = np.asarray(batches_charIds[bIdx])  ## (batchsize, 36, 11)

        batch_entity1 = np.asarray(batches_entity1_tags[bIdx]) ## (batchsize, 36)
        batch_entity1_ids = np.asarray(batches_entity1_tags_ids[bIdx]) ## (batchsize, 36)
        batch_k1 = np.asarray(batches_k1s[bIdx]) #(batchsize, )
        batch_k2 = np.asarray(batches_k2s[bIdx]) #(batchsize, )
        batch_entity2 = np.asarray(batches_entity2_tags[bIdx]) ## (batchsize, 36)
        batch_entity2_ids = np.asarray(batches_entity2_tags_ids[bIdx]) ## (batchsize, 36)
        batch_token = np.asarray(batches_tokens[bIdx]) ## (batchsize, 36)

        batch_tokenId = np.asarray(batches_tokenIds[bIdx]) ## (batchsize, 36)

        batch_doc_id = np.asarray(batches_doc_ids[bIdx]) ## (batchsize, )
 
        docs_length = np.asarray(lenBatchesDoc[bIdx])  ## (batchsize, )
        tokenslength = np.asarray(lenBatchesChars[bIdx]) ## (batchsize, 36)

        # print(np.shape(batch_embeddingIds))  
        # print(np.shape(batch_charIds))
        # print(np.shape(batch_entity1))
        # print(np.shape(batch_entity1_ids))
        # print(np.shape(batch_k1))
        # print(np.shape(batch_k2))
        # print(np.shape(batch_entity2))
        # print(np.shape(batch_entity2_ids))
        # print(np.shape(batch_token))
        # print(np.shape(batch_bio))
        # print(np.shape(batch_tokenId))
        # print(np.shape(batch_doc_id))
        # print(np.shape(docs_length))
        # print(np.shape(tokenslength))
        if train == True:
            yield {dropout_embedding_keep:dropout_embedding_prob,
               dropout_lstm_keep:dropout_lstm_prob,
               dropout_lstm_output_keep:dropout_lstm_output_prob,
               dropout_fcl_ner_keep:dropout_fcl_ner_prob,
               # isTrain:train,
               charIds:batch_charIds,
               tokensLens:tokenslength, 
               embeddingIds: batch_embeddingIds, 
               entity1_tags_ids:batch_entity1_ids,
               entity1_tags:batch_entity1, 
               k1:batch_k1, 
               k2:batch_k2,
               entity2_tags_ids:batch_entity2_ids,
               entity2_tags:batch_entity2,
               tokens:batch_token,
               tokenIds:batch_tokenId, 
               seqlen:docs_length, 
               doc_ids:batch_doc_id 
               }
        else:
            yield {dropout_embedding_keep:dropout_embedding_prob,
                   dropout_lstm_keep:dropout_lstm_prob,
                   dropout_lstm_output_keep:dropout_lstm_output_prob,
                   dropout_fcl_ner_keep:dropout_fcl_ner_prob,
                   # isTrain:train,
                   charIds:batch_charIds,
                   tokensLens:tokenslength, 
                   embeddingIds: batch_embeddingIds, 
                   entity1_tags_ids:batch_entity1_ids,
                   entity1_tags:batch_entity1, 
                   tokens:batch_token,
                   tokenIds:batch_tokenId, 
                   seqlen:docs_length, 
                   doc_ids:batch_doc_id 
               }, data_copy.data[bIdx]


def generate_eval(data, m, config, train=False):
    # generate the data
    embeddingIds = m['embeddingIds']##
    # isTrain=m['isTrain']

    # scoringMatrixGold = m['scoringMatrixGold']
    BIO = m['BIO'] # always the BIO tags
    tokens = m['tokens']
    tokenIds = m['tokenIds']
    charIds = m['charIds'] ##
    tokensLens = m['tokensLens'] ##

    seqlen = m['seqlen'] ##

    entity1_tags = m['entity1_tags'] # either the BIO tags or the EC tags - depends on the NER target values
    entity1_tags_ids = m['entity1_tags_ids']

    entity2_tags = m['entity2_tags'] # either the BIO tags or the EC tags - depends on the NER target values
    entity2_tags_ids = m['entity2_tags_ids']

    k1 = m['k1']
    k2 = m['k2']
    doc_ids=m['doc_ids']


    dropout_embedding_keep = m['dropout_embedding'] ##
    dropout_lstm_keep = m['dropout_lstm'] ##
    dropout_lstm_output_keep = m['dropout_lstm_output'] ##
    dropout_fcl_ner_keep = m['dropout_fcl_ner']

    dropout_embedding_prob = 1
    dropout_lstm_prob = 1
    dropout_lstm_output_prob = 1
    dropout_fcl_ner_prob = 1
    dropout_fcl_rel_prob = 1

    if config.use_dropout == True and train==True:

        dropout_embedding_prob = config.dropout_embedding
        dropout_lstm_prob = config.dropout_lstm
        dropout_lstm_output_prob = config.dropout_lstm_output
        dropout_fcl_ner_prob = config.dropout_fcl_ner
        dropout_fcl_rel_prob = config.dropout_fcl_rel

    data_copy = copy.deepcopy(data)
    # train_ind=np.arange(len(train.data))
    if config.shuffle == True:
        shuffled_data, _, shuffled_data_idx, _ = train_test_split(data_copy.data, data_copy.indices, test_size=0, random_state=42)
        # shuffled_data, _, shuffled_data_idx, _ = train_test_split(data_copy.data, data_copy.indices, test_size=0,random_state=42)

        data_copy = HeadData(shuffled_data, shuffled_data_idx)
        # print ("shuffle:"+ str(shuffle) )
        # print(data_copy.indices)
    else:

        data_copy = HeadData(data_copy.data, data_copy.indices)
        # data_copy = HeadData(data_copy.data, data_copy.indices)

        # print("shuffle:" + str(shuffle))
        # print(data_copy.indices)

    
    
    for docIdx in range(len(data_copy.data)):
        doc = data_copy.data[docIdx]
        char_Ids = doc.char_ids
        tokensLen= [len(token) for token in char_Ids]

        longest_token_list=max(char_Ids, key=len)
        maxTokenLen = len(longest_token_list)
        for tokenIdx in range(len(tokensLen)):
            tokenLen = tokensLen[tokenIdx]
            if tokenLen < maxTokenLen:
                for i in np.arange(maxTokenLen-tokenLen):
                    char_Ids[tokenIdx].append(0)

        # print (doc)
        # print(np.shape(np.asarray([charIds])))
        # print(np.shape(np.asarray([tokensLen])))
        # print(np.shape(np.asarray([doc.embedding_ids])))
        # print(np.shape(np.asarray([doc.tokens])))
        # print(np.shape(np.asarray([doc.BIOs])))
        # print(np.shape(np.asarray([doc.token_ids])))
        # print(np.shape(np.asarray([len(doc.embedding_ids)])))
        # print(np.shape(np.asarray([doc.docId])))


        yield {dropout_embedding_keep:dropout_embedding_prob,
               dropout_lstm_keep:dropout_lstm_prob,
               dropout_lstm_output_keep:dropout_lstm_output_prob,
               dropout_fcl_ner_keep:dropout_fcl_ner_prob,
               # isTrain:train,
               charIds:np.asarray([char_Ids]),
               tokensLens: np.asarray([tokensLen]), 
               embeddingIds: np.asarray([doc.embedding_ids]), 
               tokens: np.asarray([doc.tokens]),
               BIO: np.asarray([doc.BIOs]),
               tokenIds: np.asarray([doc.token_ids]), 
               seqlen: np.asarray([len(doc.embedding_ids)]), 
               doc_ids: np.asarray([doc.docId])
               }, doc



def getNerAndRel(doc):
    true_ners = []
    true_triples = []

    for ade in doc.ades:
        e = ade[0]
        d = ade[1]

        e_pos = doc.effect[e]
        d_pos = doc.drug[d]

        ner1 = ('E', e_pos[0], e_pos[1])
        true_ners.append(ner1)
        ner2 = ('D', d_pos[0], d_pos[1])
        true_ners.append(ner2)

        e1 = (e_pos[0], e_pos[1])
        e2 = (d_pos[0], d_pos[1])
        true_triples.append((e1, 'Ade', e2))

    return true_ners, true_triples


def getNer(seq, start, id2tag):
    tag = id2tag[seq[start]]
    [tag_class, tag_type] = tag.split('-')
    j = start + 1
    while j < len(seq):
        t = id2tag[seq[j]]
        t_class, t_type = t.split('-')[0], t.split('-')[-1]
        
        if t_type != tag_type or t_class == 'B':
            break
        j += 1
    return (tag_type, start, j-1)


def getRel(seq, rel2id, id2rel):
    rels = []
    chunks = []
    rel_default = rel2id['N']
    i = 0 
    while i < len(seq):
        idx = seq[i]
        if idx == rel_default:
            i += 1
            continue
        rel_ner = id2rel[idx]
        rel, ner = rel_ner.split("__")[0], rel_ner.split("__")[1] ## example: Kill__B-Peop
        tag_type, j = getPos(seq, i, rel, ner, id2rel)
        chunk = (tag_type, i, j)
        chunks.append(chunk)
        rels.append(rel)
        i = j + 1

    return chunks, rels


def getPos(seq, start, rel, ner, id2rel):
    tag_class, tag_type = ner.split('-')[0], ner.split('-')[-1]
    j = start+1
    while j < len(seq):
        idx = seq[j]
        rel_ner = id2rel[idx]
        if rel not in rel_ner or tag_type not in rel_ner or 'B-' in rel_ner:
            break
        j += 1
    return tag_type, j-1


def ConvertToEntity(seq, tag2id):
    # method implemented in https://github.com/guillaumegenthial/sequence_tagging/blob/master/model/data_utils.py
    """Given a sequence of tags, group entities and their position
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    id2tag = dict(zip(tag2id.values(), tag2id.keys()))

    default = tag2id['O']
    # idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i-1)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = id2tag[tok].split('-')[0], id2tag[tok].split('-')[-1]
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i-1)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq)-1)
        chunks.append(chunk)

    return chunks


def collectNerAndRel(ners1, ners2, rels):
    Ners = ners1 + ners2
    Rels = []
    for ner1 in ners1:
        e1 = (ner1[1], ner1[2])
        for i in range(len(ners2)):
            ner2 = ners2[i]
            r = rels[i]
            e2 = (ner2[1], ner2[2])
            if e1 == e2:
                continue
            Rels.append((e1, r, e2))
    return Ners, Rels