import math


def DCG(label_list):
    dcgsum = 0
    for i in range(len(label_list)):
        dcg = (2 ** label_list[i] - 1) / math.log(i + 2, 2)
        dcgsum += dcg
    return dcgsum


def NDCG(label_list, topk):
    dcg = DCG(label_list[0:topk])
    ideal_list = sorted(label_list, reverse=True)
    ideal_dcg = DCG(ideal_list[0:topk])
    if ideal_dcg == 0:
        return 0
    return dcg / ideal_dcg


def queryNDCG(label_qid_score, topk):
    tmp = sorted(label_qid_score, key=lambda x: -x[1])
    label_list = []
    for label, s in tmp:
        if label == -1:
            continue
        label_list.append(label)
    return NDCG(label_list, topk)


def cal_ndcg(labels, logits, topk):
    ndcg = 0
    for i in range(len(labels)):
        score = logits[i]
        label = labels[i]
        ndcg += queryNDCG([[label[i], score[i]] for i in range(label.shape[0])], topk)
    return ndcg, len(labels)