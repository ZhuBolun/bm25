import pickle
from collections import Counter
from files import porter
from evaluate import print_res
import argparse
import math

parser = argparse.ArgumentParser(description="parser")
parser.add_argument('-m',default="manual", help = "manual")
args = parser.parse_args()


class BM25_Model(object):
    def __init__(self, documents_list, documents_list_org, term2code, k1=2, k2=1, b=0.5):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
        self.f = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.documents_list_org = documents_list_org
        self.term2code = term2code
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
        #  Sort dictionaries
        new_idx_weight = dict(sorted(score_dict.items(), reverse=True, key=lambda x: x[1]))
        # get new index
        new_idx_list = list(new_idx_weight.keys())
        idxs = new_idx_list[:k]
        res = [idx+1 for idx in idxs]
        res_score = [new_idx_weight[idx] for idx in idxs]
        return res,res_score

def bm25load(fpath):
    f = open(fpath, 'rb')
    bm25Model = pickle.load(f)
    f.close()
    return bm25Model

def get_stopwords():
    stop_words = []
    with open('files/stopwords.txt', 'r', encoding='utf8') as f2:
        lines = f2.readlines()
        for line in lines:
            stop_words.append(line[:-1])
    return stop_words

def process_q(s,stop_words,p):
    words = s.strip().split()
    new_words = []
    for w in words:
        if w not in stop_words:
            new_words.append(p.stem(w))
    return new_words



def manual():
    p = porter.PorterStemmer()
    print('Loading BM25 index from file, please wait.')
    bm25_model = bm25load('bm25_weights.pkl')
    # get stop word
    stop_words = get_stopwords()
    q = input('Enter query:')

    while q!='QUIT':
        new_q = process_q(q,stop_words,p)
        recall_idxs,recall_scores = bm25_model.get_top_k_codes(new_q , k=15)
        print('Results for query [{}]'.format(q))
        for ii,(idx, score) in enumerate(zip(list(recall_idxs), recall_scores)):
            print('{} {} {}'.format(ii+1,idx,score))
        q = input('Enter query:')

def evaluation():
    print_res()


if __name__ == '__main__':
    if args.m == 'manual':
        manual()
    if args.m == 'evaluation':
        evaluation()



