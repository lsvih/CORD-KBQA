import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import datetime
import time

from att_lstm import AttLSTM
from lstm_dssm import AttLSTMDSSM
# from lstm_dssm_para import AttLSTMdssmPara
import data_helpers
import utils
from configure import FLAGS
from configure import base_path

# from sklearn.metrics import f1_score
import warnings
# import sklearn.exceptions
# warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# original version, classification
def train():
    utils.init_rel_dict()
    with tf.device('/cpu:0'):
        print('Loading train/dev/test data ...')
        x_text, y = data_helpers.load_data_and_labels(FLAGS.train_path)
        test_x_text, test_y = data_helpers.load_data_and_labels(FLAGS.test_path)

    # Build vocabulary
    # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # dimension = FLAGS.max_sentence_length
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    # x = np.array(list(vocab_processor.fit_transform(x_text)))
    # test_x = np.array(list(vocab_processor.fit_transform(test_x_text)))
    # ! Now we add all words from train and test to vocab.
    all_text = x_text + test_x_text
    vocab_processor.fit(all_text)
    x = np.array(list(vocab_processor.transform(x_text)))
    test_x = np.array(list(vocab_processor.transform(test_x_text)))
    print("Text Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("train x = {0}".format(x.shape))
    print("train y = {0}".format(y.shape))
    print("test x = {0}".format(test_x.shape))
    print("test y = {0}".format(test_y.shape))
    print("")

    # Randomly shuffle data to split into train and test(dev)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    x_train, x_dev = x_shuffled, test_x
    y_train, y_dev = y_shuffled, test_y
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = AttLSTM(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
            gvs = optimizer.compute_gradients(model.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs/"+base_path))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            if FLAGS.embedding_path:
                pretrain_W = utils.load_glove(FLAGS.embedding_path, FLAGS.embedding_dim, vocab_processor)
                sess.run(model.W_text.assign(pretrain_W))
                print("Success to load pre-trained word2vec model!\n")

            # Generate batches
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            best_acc = 0.0  # For save checkpoint(model)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # Train
                feed_dict = {
                    model.input_text: x_batch,
                    model.input_y: y_batch,
                    model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
                    model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    feed_dict = {
                        model.input_text: x_dev,
                        model.input_y: y_dev,
                        model.emb_dropout_keep_prob: 1.0,
                        model.rnn_dropout_keep_prob: 1.0,
                        model.dropout_keep_prob: 1.0
                    }
                    summaries, loss, accuracy, predictions = sess.run(
                        [dev_summary_op, model.loss, model.accuracy, model.predictions], feed_dict)
                    dev_summary_writer.add_summary(summaries, step)

                    time_str = datetime.datetime.now().isoformat()
                    # f1 = f1_score(np.argmax(y_dev, axis=1), predictions, labels=np.array(range(1, 19)), average="macro")
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    # print("[UNOFFICIAL] (2*9+1)-Way Macro-Average F1 Score (excluding Other): {:g}\n".format(f1))

                    # Model checkpoint
                    if best_acc < accuracy:
                        best_acc = accuracy
                        path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(best_acc), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))


# multi classifier, only feeding the question.
def multi_train(benchmark='WebQSP'):
    base_path = 'data/webqsp/l2_stype'
    if benchmark == 'SQ':
        base_path = 'data/sq/l2_stype'
    dir_names = os.listdir(base_path)
    # Pre-trained word2vec
    for dir_name in dir_names:
        print(dir_name)
        utils.init_rel_dict(os.path.join(base_path, dir_name, 'relations.txt'))
        train_path = os.path.join(base_path, dir_name, benchmark+'.RC.train')
        test_path = os.path.join(base_path, dir_name, benchmark+'.RC.test')
        out_dir = os.path.join(base_path.replace('data', 'runs'), dir_name)

        with tf.device('/cpu:0'):
            x_text, y = data_helpers.load_data_and_labels(train_path, typeinfo=FLAGS.allow_type)
            test_x_text, test_y = data_helpers.load_data_and_labels(test_path, typeinfo=FLAGS.allow_type)

        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
        all_text = x_text + test_x_text
        vocab_processor.fit(all_text)
        x = np.array(list(vocab_processor.transform(x_text)))
        test_x = np.array(list(vocab_processor.transform(test_x_text)))
        print("Text Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("train x = {0}".format(x.shape))
        print("train y = {0}".format(y.shape))
        print("test x = {0}".format(test_x.shape))
        print("test y = {0}".format(test_y.shape))
        print("")

        # Split train/test set
        x_train, x_dev = x, test_x
        y_train, y_dev = y, test_y
        print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                model = AttLSTM(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    hidden_size=FLAGS.hidden_size,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
                gvs = optimizer.compute_gradients(model.loss)
                capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
                train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

                # Output directory for models and summaries
                # timestamp = str(int(time.time()))
                # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs/"+base_path))
                print("Writing to {}\n".format(out_dir))

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

                # Write vocabulary
                vocab_processor.save(os.path.join(out_dir, "vocab"))

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                # Pre-trained word2vec
                if FLAGS.embedding_path:
                    pretrain_W = utils.load_glove(FLAGS.embedding_path, FLAGS.embedding_dim, vocab_processor)
                    sess.run(model.W_text.assign(pretrain_W))
                    print("Success to load pre-trained word2vec model!\n")

                # Generate batches
                batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                # Training loop. For each batch...
                best_acc = 0.0  # For save checkpoint(model)
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    # Train
                    feed_dict = {
                        model.input_text: x_batch,
                        model.input_y: y_batch,
                        model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
                        model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
                        model.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, loss, accuracy = sess.run(
                        [train_op, global_step, model.loss, model.accuracy], feed_dict)

                    # Training log display
                    if step % FLAGS.display_every == 0:
                        time_str = datetime.datetime.now().isoformat()
                        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                    # Evaluation
                    if step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        feed_dict = {
                            model.input_text: x_dev,
                            model.input_y: y_dev,
                            model.emb_dropout_keep_prob: 1.0,
                            model.rnn_dropout_keep_prob: 1.0,
                            model.dropout_keep_prob: 1.0
                        }
                        loss, accuracy, predictions = sess.run(
                            [model.loss, model.accuracy, model.predictions], feed_dict)

                        time_str = datetime.datetime.now().isoformat()
                        # f1 = f1_score(np.argmax(y_dev, axis=1), predictions, labels=np.array(range(1, 19)), average="macro")
                        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                        # print("[UNOFFICIAL] (2*9+1)-Way Macro-Average F1 Score (excluding Other): {:g}\n".format(f1))

                        # Model checkpoint
                        if best_acc < accuracy:
                            best_acc = accuracy
                            path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(best_acc), global_step=step)
                            print("Saved model checkpoint to {}\n".format(path))
        tf.reset_default_graph()

# [key] semantic matching
def train_lstm_dssm():
    utils.init_rel_dict()
    with tf.device('/cpu:0'):
        print('Loading train/dev/test data ...')
        train_q_text, train_r_text, train_label = data_helpers.load_dssm_data(FLAGS.train_path, typeinfo=FLAGS.allow_type)
        test_q_text, test_r_text, test_label = data_helpers.load_dssm_data(FLAGS.test_path, typeinfo=FLAGS.allow_type)
        # row_test_data = [data.split('\t')[0]+'\t'+' '.join(data.split('\t')[1].split(' ')[:4])+'\t'+data.split('\t')[2]
        #                  for data in utils.loadLists('data/webqsp/WebQSP.RE.stype.test')]
        row_test_data = utils.loadLists(FLAGS.test_path)

    # Build vocabulary
    print('Building vocabulary ...')
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    all_text = train_q_text + train_r_text + test_q_text + test_r_text
    vocab_processor.fit(all_text)
    train_q_ids = np.array(list(vocab_processor.transform(train_q_text)))
    train_r_ids = np.array(list(vocab_processor.transform(train_r_text)))
    train_label = np.array(train_label)
    test_q_ids = np.array(list(vocab_processor.transform(test_q_text)))
    test_r_ids = np.array(list(vocab_processor.transform(test_r_text)))
    test_label = np.array(test_label)
    print("Text Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("train x = {0}".format(train_q_ids.shape))
    print("train y = {0}".format(train_label.shape))
    print("test x = {0}".format(test_q_ids.shape))
    print("test y = {0}".format(test_label.shape))
    print("")

    # Randomly shuffle data to split into train and test(dev)
    print("Train/Dev split: {:d}/{:d}\n".format(len(train_label), len(test_label)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = AttLSTMDSSM(
                sequence_length=train_q_ids.shape[1],       # 50
                num_classes=train_label.shape[1],           # 2
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
            gvs = optimizer.compute_gradients(model.loss)
            # print(gvs)
            # gvs_ = [(0 if grad is None else grad, var) for grad, var in gvs]
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs/"+base_path, 'dssm_base_sigmoid_deep_nodrop_para'))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            results_dir = os.path.abspath(os.path.join(out_dir, "results"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            if FLAGS.embedding_path:
                pretrain_W = utils.load_glove(FLAGS.embedding_path, FLAGS.embedding_dim, vocab_processor)
                sess.run(model.W_text.assign(pretrain_W))
                print("Success to load pre-trained word2vec model!\n")

            # Generate batches
            batches = data_helpers.batch_iter(list(zip(train_q_ids, train_r_ids, train_label)),
                                              FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            best_acc = 0.0  # For save checkpoint(model)
            acc_list = []
            # for epoch in range(FLAGS.num_epochs):
            #     train_q_text, train_r_text, train_label = data_helpers.load_dssm_data(FLAGS.train_path)
            #     train_q_ids = np.array(list(vocab_processor.transform(train_q_text)))
            #     train_r_ids = np.array(list(vocab_processor.transform(train_r_text)))
            #     train_label = np.array(train_label)
            #     batches = data_helpers.batch_iter(list(zip(train_q_ids, train_r_ids, train_label)),
            #                                   FLAGS.batch_size, 1)
            for batch in batches:
                q_batch, r_batch, y_batch = zip(*batch)
                # Train
                feed_dict = {
                    model.input_q: q_batch,
                    model.input_r: r_batch,
                    model.input_y: y_batch,
                    model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
                    model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss = sess.run(
                    [train_op, global_step, model.loss], feed_dict)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}".format(time_str, step, loss))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    test_batches = data_helpers.batch_iter(list(zip(test_q_ids, test_r_ids, test_label)),
                                              FLAGS.batch_size, 1, shuffle=False)
                    preds = []
                    for test_batch in test_batches:
                        test_q_batch, test_r_batch, test_y_batch = zip(*test_batch)
                        feed_dict = {
                            model.input_q: test_q_batch,
                            model.input_r: test_r_batch,
                            model.input_y: test_y_batch,
                            model.emb_dropout_keep_prob: 1.0,
                            model.rnn_dropout_keep_prob: 1.0,
                            model.dropout_keep_prob: 1.0
                        }
                        logits = sess.run([model.logits], feed_dict)
                        logits = [r[0] for r in logits[0]]
                        preds += logits
                    time_str = datetime.datetime.now().isoformat()
                    acc = utils.evaluate_rel_accuracy(preds, row_test_data)
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))

                    # Model checkpoint
                    if best_acc < acc:
                        best_acc = acc
                        acc_list.append(acc)
                        path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(best_acc), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))
                        utils.writeList(os.path.join(results_dir, str(step)+'.out'), preds)
                        utils.writeList(os.path.join(results_dir, 'accList.txt'), acc_list)


def main(_):
    # train()
    train_lstm_dssm()

if __name__ == "__main__":
    # multi_train()
    train_lstm_dssm()
    # tf.app.run()
