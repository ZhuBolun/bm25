import math
from files import porter
import pickle
def get_std_res():
    res = {}
    weights_dict = {}
    with open('files/qrels.txt','r',encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            qid = line.split()[0]
            recall_f = line.split()[2]
            # Get wright
            weight = int(line.split()[3])
            if qid not in res.keys():
                res[qid] = [recall_f]
            else:
                res[qid].append(recall_f)
            if qid not in weights_dict.keys():
                weights_dict[qid] = {}
            weights_dict[qid][recall_f] = weight

    return res,weights_dict

def get_predict_res():
    res = {}
    with open('output.txt','r',encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            qid = line.split()[0]
            recall_f = line.split()[2]
            if qid not in res.keys():
                res[qid] = [recall_f]
            else:
                res[qid].append(recall_f)
    return res

def get_quires():
    fn = 'files/queries.txt'
    quires = []
    with open(fn,'r',encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            new_line = ' '.join(line.split()[1:])
            quires.append(new_line)
    return quires

def bm25load(fpath):
    f = open(fpath, 'rb')
    bm25Model = pickle.load(f)
    f.close()
    return bm25Model


def save_to_output(quires,stop_words,fn='output.txt'):
    p = porter.PorterStemmer()
    # loading BM25
    print('Loading BM25 index from file, please wait.\n')
    bm25_model = bm25load('bm25_weights.pkl')
    with open(fn, 'w', encoding='utf8') as f:
        i=1
        for q in quires:
            # q = 'what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft'
            new_q = process_q(q,stop_words,p)
            # recall 50 word
            recall_codes,recall_scores = bm25_model.get_top_k_codes(new_q, k=50)
            j=1
            for c,s in zip(recall_codes,recall_scores):
                f.write('{} {} {} {} {} {}\n'.format(i,'Q0',c,j,round(s,4),'19206215'))
                j+=1
            i += 1

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


def precision(std,predict):
    # 50 doc
    pres = []
    for qid,recall_files in predict.items():
        recall_true = len(set(recall_files) & set(std[qid]))
        pres.append(float(recall_true/len(recall_files)))
    print("Precision:".ljust(18, " "), sum(pres)/len(std))

def recall(std,predict):
    rs = []
    for qid, recall_files in predict.items():
        recall_true = len(set(recall_files) & set(std[qid]))
        rs.append(float(recall_true / len(std[qid])))
    print("Recall: ".ljust(18," "),sum(rs)/len(std))

def MAP(std,predict):
    pos_dict = {}
    for qid,recall_files in predict.items():
        for i,idx in enumerate(recall_files):
            if idx in std[qid]:
                if qid not in pos_dict.keys():
                    # get the rank
                    pos_dict[qid] = [i+1]
                else:
                    pos_dict[qid].append(i+1)
    sum_map = 0
    for qid, idxs in pos_dict.items():
        for i in range(1,len(idxs)+1):
            one_map = float(i/idxs[i-1])/len(std[qid])
            sum_map+=one_map
    ave_map = sum_map/len(std)

    print("MAP:".ljust(18," "),ave_map)
    return None


def R_precision(std,predict):
    pres = []
    for qid, recall_files in predict.items():
        recall_true = len(set(recall_files[:len(std[qid])]) & set(std[qid]))
        pres.append(float(recall_true / len(std[qid])))
    print("R-precision:".ljust(18," "),sum(pres)/len(std))

def p10(std,predict):
    pres = []
    for qid, recall_files in predict.items():
        recall_true = len(set(recall_files[:10]) & set(std[qid]))
        pres.append(float(recall_true / len(recall_files[:10])))
    print("P@10: ".ljust(18," "),sum(pres)/len(std))

def Bpref(std,predict):
    def cal_pre_norevalent(pos,new_pos):
        one_p = 0
        for p in pos:
            if p<=len(pos):
                one_p += (1 - float((new_pos[:p].count(0)/len(pos))))
        return 1/len(pos)*one_p

    pos_dict = {}
    for qid,recall_files in predict.items():
        for i,idx in enumerate(recall_files):
            if idx in std[qid]:
                if qid not in pos_dict.keys():
                    # record the rank
                    pos_dict[qid] = [i+1]
                else:
                    pos_dict[qid].append(i+1)
    sum_pref = 0
    for qid,pos in pos_dict.items():
        # add '0'
        new_pos = [0]*pos[-1]
        for p in pos:
            new_pos[p-1] = p
        sum_pref+=cal_pre_norevalent(pos,new_pos)

    ave_pref = float(sum_pref/len(std))
    print("bpref: ".ljust(18," "),ave_pref)

    return None

def NDCG(std,predict,weights_dict):
    values_dict ={1:1,2:3,3:8,4:15}
    def discount_factor(rank):
        return float(1/math.log(rank))
    res_weight_dict = {}
    for qid, recall_files in predict.items():
        for i, idx in enumerate(recall_files):
            if idx in std[qid]:
                if qid not in res_weight_dict.keys():
                    res_weight_dict[qid] = {}
                res_weight_dict[qid][idx] = weights_dict[qid][idx]
    DCG = {}
    for qid,idx_weight in res_weight_dict.items():
        last_p = 0
        for i,(indx,weight) in enumerate(idx_weight.items()):
            if qid not in DCG.keys():
                DCG[qid] = {}
            DCG[qid][indx] = last_p+values_dict[weight]*discount_factor(i+2)
            last_p = last_p+values_dict[weight]*discount_factor(i+2)
    M_DCG = {}
    # In reverse order by value
    for qid, idx_weight in res_weight_dict.items():
        # Sort in reverse order by key value
        last_p = 0
        new_idx_weight = dict(sorted(idx_weight.items(),reverse=True, key=lambda x: x[1]))
        for i, (indx, weight) in enumerate(new_idx_weight.items()):
            if qid not in M_DCG.keys():
                M_DCG[qid] = {}
            M_DCG[qid][indx] = last_p+values_dict[weight]*discount_factor(i+2)
            last_p = last_p+values_dict[weight]*discount_factor(i+2)
    # normalization
    N_DCG = {}
    for qid,idx_weight in DCG.items():
        for idx, weight in idx_weight.items():
            if qid not in N_DCG.keys():
                N_DCG[qid] = []
            nor_data = float(DCG[qid][idx]/M_DCG[qid][idx])
            N_DCG[qid].append(nor_data)
    # average
    N_DCG_sum = 0
    for qid,values in N_DCG.items():
        N_DCG_sum+=float(sum(values)/len(values))

    print("NDCG:".ljust(18," "),N_DCG_sum/len(std))

def print_res():
    quires = get_quires()
    stop_words = get_stopwords()
    save_to_output(quires, stop_words,'output.txt')
    print('Evaluation results: ')
    std_res, weights_dict = get_std_res()
    predict_res = get_predict_res()
    # Precision:
    precision(std_res, predict_res)
    # Recall：
    recall(std_res, predict_res)
    # P@10
    p10(std_res, predict_res)
    # R-precision
    R_precision(std_res, predict_res)
    # map
    MAP(std_res, predict_res)
    # bpref
    Bpref(std_res, predict_res)
    # NDCG
    NDCG(std_res, predict_res, weights_dict)






