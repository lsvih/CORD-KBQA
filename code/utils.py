import numpy as np
import codecs
import tensorflow as tf
from configure import base_path

rel2id = {}
id2rel = {}
id2ques = {}
rid2pqs = {}

# input
def loadLists(filename, retTypeSet=False):
    inputLines = codecs.open(filename, encoding='utf-8').readlines()
    retList = [line.strip() for line in inputLines]
    if retTypeSet:
        retList = set(retList)
    return retList

# output
def writeList(filename, listName, convert=str):
    output = codecs.open(filename, "w", "utf-8")
    for li in listName:
        output.write(convert(li) + '\n')
    output.flush()
    output.close()


def init_rel_dict(rel_dict_path=''):
    rel2id.clear()
    id2rel.clear()
    if rel_dict_path == '':
        rel_dict_path = './data/'+base_path+'relations.txt'
    rels = [line.strip() for line in open(rel_dict_path)]
    for rel in rels:
        rel2id[rel] = len(rel2id)
        id2rel[len(id2rel)] = rel
    print('Load rel dict done.')
    # load parallel questions for relations (training data)
    min_pq_num = 5
    rel_cnt_path = './data/'+base_path+'train.rels'
    train_path = './data/'+base_path+'WebQSP.RE.stype.train'
    qid = 0
    for line in loadLists(train_path):
        id2ques[qid] = line.split('\t')[2]
        qid += 1
    for line in loadLists(rel_cnt_path):
        rid = line.split('\t')[0]
        pqids = line.split('\t')[2]
        pqid_list = pqids[1:-1].split(', ')
        pqs = [id2ques[int(pqid)] for pqid in pqid_list]
        if len(pqs) >= min_pq_num:
            rid2pqs[rid] = pqs


# arg1: prediction scores of all candidate (q,r) pairs
def evaluate_rel_accuracy(scores, goldData, logOutputFile=None):
    relCnt = 0
    for data in goldData:
        relCnt += 1
        if not data.split('\t')[1].startswith('n'):
            relCnt += len(data.split('\t')[1].split(' '))
    print('predicted scores len: ' + str(len(scores)) + '\tgold data len: ' + str(relCnt))
    assert len(scores) == relCnt

    # unseen_rels = [x.split('\t')[0] for x in loadLists('qa/data/webqsp/test_unseen.rels')]
    unseen_rels = []
    accList = []
    errList = []
    allList = []
    accNum = 0
    pos = 0
    for data in goldData:
        rs = {}
        ques = data.split('\t')[2]
        goldrid = data.split('\t')[0]
        rs[goldrid] = scores[pos]
        pos += 1
        if not data.split('\t')[1].startswith('n'):
            for rid in data.split('\t')[1].split(' '):
                rs[rid] = scores[pos]
                pos += 1
        srs = sorted(rs.items(), key=lambda x: x[1], reverse=True)
        score_data = [rs[0] + ':' + str(rs[1]) for rs in srs]
        rank = 0
        lastScore = 0
        for rs in srs:
            if rs[1] <= lastScore:  # NOTICE: Not allow share the best score with WRONG relations.
                rank += 1
            lastScore = rs[1]
            flag = False
            for rid in rs[0].split():
                if rid in goldrid.split():
                    flag = True
                    break
            if flag:
                break

        # top 1 & not all zero score
        if rank == 0 and srs[0][1] != 0:
            accNum += 1
            accList.append(data)
        # elif goldrid in unseen_rels:    # unseen rels err
        #     errList.append(data)
        else:   # gold | predict | ques
            errList.append(goldrid + '\t' + srs[0][0] + '\t' + ques)

        allList.append(str(rank) + '\t' + str(score_data) + '\t' + ques)
    errList = sorted(errList)
    if logOutputFile:
        writeList(logOutputFile, errList)
    print('acc/total %d/%d : rate %f' % (accNum, len(goldData), float(accNum)/len(goldData)))
    return float(accNum)/len(goldData)


def gen_rel_classification_data():
    init_rel_dict()
    data_path_list = ['./data/webqsp/WebQSP.RE.stype.train', './data/webqsp/WebQSP.RE.stype.test']
    rel_set = set()
    for data_path in data_path_list:
        datas = loadLists(data_path)
        outputs = []
        output_path = './data/webqsp/WebQSP.RC.train'
        if 'test' in data_path:
            output_path = output_path.replace('train', 'test')
        line_id = 0
        for data in datas:
            line_id += 1
            gold_rid = data.split('\t')[0].split(' ')[0]
            gold_rel = id2rel[int(gold_rid)-1]
            rel_set.add(gold_rel)
            ques = data.split('\t')[2]
            outputs.append(str(line_id)+'\t'+ques)
            outputs.append(gold_rel+'\n\n')
        writeList(output_path, outputs)
    writeList('./data/webqsp/positive_relations.txt', list(rel_set))

def load_glove(embedding_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) / np.sqrt(len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load glove file {0}".format(embedding_path))
    f = open(embedding_path, 'r', encoding='utf8')
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        idx = vocab.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = embedding
    return initW


def tf_vocab_test():
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(10)
    x = np.array(list(vocab_processor.fit_transform(['this is <e> test', 'while \'s ok'])))
    print(x)
    test_x = np.array(list(vocab_processor.fit_transform(['no word', 'give me <e> unk'])))
    print(test_x)

if __name__ == '__main__':
    gen_rel_classification_data()
    # tf_vocab_test()
