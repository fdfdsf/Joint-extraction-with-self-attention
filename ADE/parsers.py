import utils
import csv
import pandas as pd
import numpy as np
import re
import json


class Doc:
    def __init__(self, id):
        ## append
        self.docId = id
        self.text = ''
        self.tokens = []
        self.ades = []
        self.effect = {} ## {'word':()}
        self.drug = {}  ## {'word':()}

        ### extend
        self.token_ids = []
        self.char_ids = []
        self.BIO1s = []
        self.BIO1_ids = []
        self.BIO2s = []
        self.BIO2_ids = []


    def append(self, text, tokens, ades):
        self.text = text
        self.tokens = tokens
        self.ades = ades


    def extend(self, wordindices, dataset_set_characters, dataset_set_bio_tags, dataset_set_relations, dataset_set_bio_relation_ners):
        self.BIO1s, self.effect, self.drug = utils.getPosAndBio1s(self.text, self.tokens, self.ades)
        for tId in range(len(self.tokens)):
            self.token_ids.append(int(utils.getEmbeddingId(self.tokens[tId], wordindices)))
            self.char_ids.append(utils.tokenToCharIds(self.tokens[tId], dataset_set_characters))
            self.BIO1_ids.append(int(utils.getLabelId(self.BIO1s[tId], dataset_set_bio_tags)))


def readHeadFile(headFile):
    # head_id_col_vector = ['tId', 'emId', "token", "nerId", "nerBilou","nerBIO", "ner", 'relLabels', "headIds", 'rels', 'relIds','scoringMatrixHeads','tokenWeights']
    docs = []
    did = 1
    with open(headFile, 'r', encoding='utf-8') as f:
        for line in f:
            doc = Doc(str(did))
            data = json.loads(line.strip())
            text = data['Text']
            tokens = text.split(' ')
            ades = data['Relations']
            doc.append(text, tokens, ades)
            docs.append(doc)
            did += 1

    return docs


def preprocess(docs, wordindices, dataset_set_characters, dataset_set_bio_tags, dataset_set_relations, dataset_set_bio_relation_ners):
    for doc in docs:
        doc.extend(wordindices, dataset_set_characters, dataset_set_bio_tags, dataset_set_relations, dataset_set_bio_relation_ners)


class read_properties:
    def __init__(self,filepath, sep='=', comment_char='#'):
        """Read the file passed as parameter as a properties file."""
        self.props = {}
        #print filepath
        with open(filepath, "rt") as f:
            for line in f:
                #print line
                l = line.strip()
                if l and not l.startswith(comment_char):
                    key_value = l.split(sep)
                    self.props[key_value[0].strip()] = key_value[1].split("#")[0].strip('" \t')


    def getProperty(self,propertyName):
        return self.props.get(propertyName)
