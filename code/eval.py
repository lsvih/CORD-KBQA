import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import subprocess

import data_helpers
import utils
from configure import FLAGS
from utils import loadLists, writeList


# for RC classifier
def eval():
    utils.init_rel_dict()
    with tf.device('/cpu:0'):
        x_text, y = data_helpers.load_data_and_labels(FLAGS.test_path)

    # Map data into vocabulary
    text_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    x = np.array(list(text_vocab_processor.transform(x_text)))

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            emb_dropout_keep_prob = graph.get_operation_by_name("emb_dropout_keep_prob").outputs[0]
            rnn_dropout_keep_prob = graph.get_operation_by_name("rnn_dropout_keep_prob").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            preds = []
            for x_batch in batches:
                pred = sess.run(predictions, {input_text: x_batch,
                                              emb_dropout_keep_prob: 1.0,
                                              rnn_dropout_keep_prob: 1.0,
                                              dropout_keep_prob: 1.0})
                preds.append(pred)
            preds = np.concatenate(preds)
            truths = np.argmax(y, axis=1)

            prediction_path = os.path.join(FLAGS.checkpoint_dir, "..", "predictions.txt")
            truth_path = os.path.join(FLAGS.checkpoint_dir, "..", "ground_truths.txt")
            prediction_file = open(prediction_path, 'w')
            truth_file = open(truth_path, 'w')
            for i in range(len(preds)):
                prediction_file.write("{}\t{}\n".format(i, utils.id2rel[preds[i]]))
                truth_file.write("{}\t{}\n".format(i, utils.rel2id[truths[i]]))
            prediction_file.close()
            truth_file.close()

            '''
            perl_path = os.path.join(os.path.curdir,
                                     "SemEval2010_task8_all_data",
                                     "SemEval2010_task8_scorer-v1.2",
                                     "semeval2010_task8_scorer-v1.2.pl")
            process = subprocess.Popen(["perl", perl_path, prediction_path, truth_path], stdout=subprocess.PIPE)
            for line in str(process.communicate()[0].decode("utf-8")).split("\\n"):
                print(line)
            '''

def extract_ids(data_path):
    datas = utils.loadLists(data_path)
    ids = []
    for data in datas:
        if len(data.split('\t')) > 1:
            ids.append(data.split('\t')[0])
    return ids

def load_global_rel_dict(benchmark='WebQSP'):
    rel_dict_path = './data/webqsp/relations.txt'
    if benchmark == 'SQ':
        rel_dict_path = './data/sq/relations.txt'
    rels = [line.strip() for line in open(rel_dict_path)]
    id2rel = {}
    rel2id = {}
    for rel in rels:
        rel2id[rel] = len(rel2id)
        id2rel[len(id2rel)] = rel
    return rel2id

# for RC multi classifier
def multi_eval(benchmark='WebQSP'):
    base_path = 'data/webqsp/l2'
    if benchmark == 'SQ':
        base_path = 'data/sq/l2_rtype'
    global_rel2id = load_global_rel_dict(benchmark)
    dir_names = os.listdir(base_path)
    for dir_name in dir_names:
        print(dir_name)
        utils.init_rel_dict(os.path.join(base_path, dir_name, 'relations.txt'))
        test_path = os.path.join(base_path, dir_name, benchmark+'.RC.test')
        checkpoint_dir = os.path.join(base_path.replace('data', 'runs'), dir_name, 'checkpoints')

        with tf.device('/cpu:0'):
            x_text, y = data_helpers.load_data_and_labels(test_path)
            ids = extract_ids(test_path)

        # Map data into vocabulary
        text_path = os.path.join(checkpoint_dir, "..", "vocab")
        text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
        x = np.array(list(text_vocab_processor.transform(x_text)))

        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_text = graph.get_operation_by_name("input_text").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                emb_dropout_keep_prob = graph.get_operation_by_name("emb_dropout_keep_prob").outputs[0]
                rnn_dropout_keep_prob = graph.get_operation_by_name("rnn_dropout_keep_prob").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                batches = data_helpers.batch_iter(list(x), FLAGS.batch_size, 1, shuffle=False)

                # Collect the predictions here
                preds = []
                for x_batch in batches:
                    pred = sess.run(predictions, {input_text: x_batch,
                                                  emb_dropout_keep_prob: 1.0,
                                                  rnn_dropout_keep_prob: 1.0,
                                                  dropout_keep_prob: 1.0})
                    preds.append(pred)
                preds = np.concatenate(preds)
                truths = np.argmax(y, axis=1)

                '''
                Now the data contains following cases (multi-gold):
                354	what did <e> wrote	book.author.book_editions_published
                354	what did <e> wrote	book.author.works_written
                That leads to the accurate drops (as we only can get ONE good of the above TWO)
                So we need 1) clean the data OR 2) re-compute the accuracy
                '''
                prediction_path = os.path.join(checkpoint_dir, "..", "predictions.txt")
                truth_path = os.path.join(checkpoint_dir, "..", "ground_truths.txt")
                corrected_path = os.path.join(checkpoint_dir, "..", "corrected.txt")
                corrected = []
                prediction_file = open(prediction_path, 'w')
                truth_file = open(truth_path, 'w')
                for i in range(len(preds)):
                    prediction_file.write("{}\t{}\t{}\t{}\t{}\n".format(ids[i], x_text[i], str(global_rel2id[utils.id2rel[preds[i]]]+1), utils.id2rel[preds[i]], '1' if preds[i]==truths[i] else '0'))
                    truth_file.write("{}\t{}\t{}\t{}\n".format(ids[i], x_text[i], utils.id2rel[truths[i]], '1' if preds[i]==truths[i] else '0'))
                    if preds[i]==truths[i] and ids[i]+'\t'+x_text[i] not in corrected:
                        corrected.append(ids[i]+'\t'+x_text[i])
                real_acc = float(len(corrected))/len(set(ids))
                corrected.append(str(real_acc))
                utils.writeList(corrected_path, corrected)
                prediction_file.close()
                truth_file.close()


def eval_dssm():
    paras = {}
    result_path = './runs/webqsp/dssm_base_sigmoid_deep_nodrop_para/results/'
    test_results = [float(x) for x in loadLists(result_path + '51000.out')]
    paras['rowTestData'] = loadLists('data/webqsp/WebQSP.RE.stype.test')
    # paras['replaceTestData'] = loadLists('data/webqsp/WebQSP.RE.replace.test')
    print('evaluating ...')
    utils.evaluate_rel_accuracy(test_results, paras['rowTestData'], result_path+'51000.err')


def main(_):
    eval()

if __name__ == "__main__":
    eval_dssm()
    # multi_eval()
    # multi_eval('SQ')
    # tf.app.run()