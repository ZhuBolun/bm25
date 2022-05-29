from collections import Counter
import math
import os
from files import porter
import pickle
import string


class BM25_Model(object):
    def __init__(self, documents_list, documents_list_org, k1=2, k2=1, b=0.5):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
        self.f = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        # Original unparticipled text
        self.documents_list_org = documents_list_org
        # Get a dictionary of terms to codes
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():

            self.idf[key] = math.log((self.documents_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                        self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                                 qf[q] * (self.k2 + 1) / (qf[q] + self.k2))

        return score

    def get_documents_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list


    def get_top_k_codes(self,query, k):
        score_list = self.get_documents_score(query)
        score_dict = {}
        for i,score in enumerate(score_list):
            score_dict[i] = score
        new_idx_weight = dict(sorted(score_dict.items(), reverse=True, key=lambda x: x[1]))
        # Get new index
        new_idx_list = list(new_idx_weight.keys())
        idxs = new_idx_list[:k]
        res = [idx+1 for idx in idxs]
        res_score = [new_idx_weight[idx] for idx in idxs]
        return res,res_score



def get_essat2code():
    fs = []
    for root, dirs, files in os.walk('documents'):
        i=0
        for f in files:
            with open('documents/{}'.format(f), 'r', encoding='utf8') as f2:
                lines = f2.readlines()
                for line in lines:
                    fs.append(line)
            i+=1
    return fs

def get_stopwords():
    stop_words = []
    with open('files/stopwords.txt', 'r', encoding='utf8') as f2:
        lines = f2.readlines()
        for line in lines:
            stop_words.append(line[:-1])
    return stop_words



def get_term_list_22(stop_words,p):
    doc_dict = {}
    term_list = []

    for root,dirs,files in os.walk('documents'):
        i = 1
        for f in files:
            with open('documents/{}'.format(f),'r',encoding='utf8') as f2:
                lines = f2.readlines()
                if len(lines)==0:
                    doc_dict[f] = ['blank content\n']
                else:
                    doc_dict[f] = lines
    # Get index
    doc_list = []
    for i in range(1,len(doc_dict)+1):
        doc_list.append(doc_dict[str(i)][0])
    for i,doc in enumerate(doc_list):
        # Case conversion
        low_doc = doc.lower()
        # Remove punctuation marks
        remove = str.maketrans('', '', string.punctuation)
        without_punctuation_doc = low_doc.translate(remove)
        # Divide word
        words = without_punctuation_doc.split()
        # Remove stop word
        new_words = []
        for w in words:
            if w not in stop_words:
               new_words.append(p.stem(w))
        term_list.append(new_words)
    return term_list




def process_q(s):
    words = s.strip().split()
    new_words = []
    for w in words:
        if w not in stop_words:
            new_words.append(p.stem(w))
    return new_words

def bm25save(obj, fpath):
    f = open(fpath, 'wb')
    pickle.dump(obj, f)
    f.close()

def get_quires():
    fn = 'files/queries.txt'
    quires = []
    with open(fn,'r',encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            new_line = ' '.join(line.split()[1:])
            quires.append(new_line)
    return quires

def save_to_output(quires, fn='output.txt'):
    with open(fn, 'w', encoding='utf8') as f:
        i=1
        for q in quires:
            # q = 'what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft'
            new_q = process_q(q)
            # Recall 50 numbers
            recall_codes, recall_scores = bm25_model.get_top_k_codes(new_q, k=50)
            j = 1
            for c, s in zip(recall_codes,recall_scores):
                f.write('{} {} {} {} {} {}\n'.format(i, 'Q0', c, j, round(s, 4), '19206215'))
                j += 1
            print(i)
            i += 1



if __name__ == '__main__':
    p = porter.PorterStemmer()
    # Get stop word
    stop_words = get_stopwords()
    files = get_essat2code()
    # Get list and remove stop word
    term_list = get_term_list_22(stop_words,p)

    bm25_model = BM25_Model(term_list, files)
    bm25save(bm25_model, 'bm25_weights.pkl')




