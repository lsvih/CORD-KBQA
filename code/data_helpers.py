import numpy as np
import pandas as pd
import random
import nltk
import re

import utils
from configure import FLAGS

max_ques_len = 0
max_rel_len = 0

def clean_str(text):
    text = text.lower()
    # Clean the text
    # text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    # text = re.sub(r"\'s", " 's", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def load_data_and_labels(path, typeinfo=False):
    data = []
    lines = [line.strip() for line in open(path)]
    max_sentence_length = 60
    for idx in range(0, len(lines), 4):
        id = lines[idx].split("\t")[0]
        relation = lines[idx + 1]
        type = lines[idx + 2]
        sentence = lines[idx].split("\t")[1]  # "" -> [1:-1]
        # sentence = sentence.replace('<e1>', ' _e11_ ')
        # sentence = sentence.replace('</e1>', ' _e12_ ')
        # sentence = sentence.replace('<e2>', ' _e21_ ')
        # sentence = sentence.replace('</e2>', ' _e22_ ')

        sentence = clean_str(sentence)
        # tokens = nltk.word_tokenize(sentence)
        # if max_sentence_length < len(tokens):
        #     max_sentence_length = len(tokens)
        # sentence = " ".join(tokens)

        data.append([id, sentence, relation, type])

    print(path)
    print("max sentence length = {}\n".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence", "relation", "type"])
    df['label'] = [utils.rel2id[r] for r in df['relation']]

    # Text Data
    x_text = df['sentence'].tolist()
    x_type = df['type'].tolist()

    def padding(words, pad_str, size):
        while len(words) < size:
            words.append(pad_str)
        if len(words) > size:
            words = words[:size]
        return ' '.join(words)

    if typeinfo:
        x_type_text = []
        for type, ques in zip(x_type, x_text):
            type_words = re.split("/|\.|_", type)
            type_words = [x for x in type_words if x]
            ques_words = ques.split(' ')
            padded_type = padding(type_words, '_pad_', 10)  # type sequence limit size: 10
            padded_ques = padding(ques_words, '_pad_', max_sentence_length-10)
            x_type_text.append(padded_type + ' ' + padded_ques)
        x_text = x_type_text

    # Label Data
    y = df['label']
    labels_flat = y.values.ravel()
    # labels_count = np.unique(labels_flat).shape[0]
    labels_count = len(utils.rel2id)

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # ...
    # 18 => [0 0 0 0 0 ... 0 0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    return x_text, labels


def translate_line_to_samples_with_possibility(line):
    gold_rid = line.split('\t')[0].split(' ')[0]  # webqsp have some questions with more than one correct relations
    rid_pool = line.split('\t')[1]  # For train data this is negative pool, For test this is the whole pool
    # if len(line.split('\t'))<3:
    #     print('Err: ' + line)
    ques = line.split('\t')[2]
    samples = [[], [], []]
    tmp = []
    cands = [gold_rid]
    if not rid_pool.startswith('n'):
        tmp = rid_pool.split(' ')
    shuffle_indices = np.random.permutation(np.arange(len(tmp)))
    for x in shuffle_indices:
        cands.append(tmp[x])
        if (len(cands)-1) * 20 >= len(tmp):
            break
    for rid in cands:
        qu_words = ques.split(' ')
        rel = utils.id2rel[int(rid)-1]
        pre_words = re.split("/|\.|_", rel)
        pre_words = [x for x in pre_words if x]
        samples[0].append(ques)
        samples[1].append(' '.join(pre_words))
        samples[2].append([1, 0] if rid == gold_rid else [0, 1])
    return samples[0], samples[1], samples[2]


def translate_line_to_samples(line, typeinfo=False, parainfo=False):
    def padding(words, pad_str, size):
        while len(words) < size:
            words.append(pad_str)
        if len(words) > size:
            words = words[:size]
        return ' '.join(words)

    gold_rid = line.split('\t')[0].split(' ')[0]  # webqsp have some questions with more than one correct relations
    rid_pool = line.split('\t')[1]  # For train data this is negative pool, For test this is the whole pool
    # if len(line.split('\t'))<3:
    #     print('Err: ' + line)
    ques = line.split('\t')[2]
    # count max len
    global max_ques_len
    max_ques_len = max(max_ques_len, len(ques.split(' ')))

    type = line.split('\t')[3] if len(line.split('\t')) > 3 else ''
    type_words = re.split("/|\.|_", type)
    type_words = [x for x in type_words if x]
    padded_type = padding(type_words, '_pad_', 10)  # type sequence limit size: 10
    samples = [[], [], []]
    cands = [gold_rid]
    if not rid_pool.startswith('n'):
        cands += rid_pool.split(' ')
    i = 0
    for rid in cands:
        i += 1
        rel = utils.id2rel[int(rid)-1]
        pre_words = re.split("/|\.|_", rel)
        pre_words = [x for x in pre_words if x]
        # count max len
        global max_rel_len
        max_rel_len = max(max_rel_len, len(pre_words))
        # get parallel questions of gold relation
        pq = random.choice(utils.rid2pqs[rid]) if rid in utils.rid2pqs.keys() else ''
        padded_pq = padding(pq.split(' '), '_pad_', 30)  # parallel question sequence limit size: 30

        if typeinfo:
            samples[0].append(padded_type + ' ' + ques)
        else:
            samples[0].append(ques)
        if parainfo:
            samples[1].append(padded_pq + ' ' + ' '.join(pre_words))
        else:
            samples[1].append(' '.join(pre_words))
        samples[2].append([1, 0] if i == 1 else [0, 1])

    return samples[0], samples[1], samples[2]

def load_dssm_data(path, typeinfo=False, parainfo=False, sampling=False):
    # print('Reading: ' + path)
    np.random.seed(10)
    datas = [[], [], []]
    lines = utils.loadLists(path)
    for line in lines:
        if sampling:
            qs, rs, ls = translate_line_to_samples_with_possibility(line)   # TODO: is that need typeinfo/parainfo ?
        else:
            qs, rs, ls = translate_line_to_samples(line, typeinfo, parainfo)
        datas[0] += qs
        datas[1] += rs
        datas[2] += ls
    print('max ques len: ' + str(max_ques_len))
    print('max rel len: ' + str(max_rel_len))
    return datas[0], datas[1], datas[2]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == "__main__":
    testFile = 'data/webqsp/WebQSP.RE.stype.test'
    utils.init_rel_dict('data/webqsp/relations.txt')
    qs, rs, ls = load_dssm_data(testFile, typeinfo=False, parainfo=True)
    for i in range(0, len(qs)):
        print(qs[i])
        print(rs[i])
        print(ls[i])
