import tensorflow as tf
from attention import attention
from configure import use_attention

class AttLSTMDSSM:
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 hidden_size, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_q = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_q')  # question
        self.input_r = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_r')  # relation
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')    # label
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        initializer = tf.keras.initializers.glorot_normal

        # Word Embedding Layer
        with tf.device('/cpu:0'), tf.variable_scope("word-embeddings"):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.25, 0.25), name="W_text")
            self.embedded_q = tf.nn.embedding_lookup(self.W_text, self.input_q)
            self.embedded_r = tf.nn.embedding_lookup(self.W_text, self.input_r)

        # Dropout for Word Embedding
        with tf.variable_scope('dropout-embeddings'):
            self.embedded_q = tf.nn.dropout(self.embedded_q, self.emb_dropout_keep_prob)
            self.embedded_r = tf.nn.dropout(self.embedded_r, self.emb_dropout_keep_prob)

        # Bidirectional LSTM
        with tf.variable_scope("bi-lstm-q"):
            # Question
            _fw_cell_1 = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            fw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(_fw_cell_1, self.rnn_dropout_keep_prob)
            _bw_cell_1 = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            bw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(_bw_cell_1, self.rnn_dropout_keep_prob)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell_1,
                                                                  cell_bw=bw_cell_1,
                                                                  inputs=self.embedded_q,
                                                                  sequence_length=self._length(self.input_q),
                                                                  dtype=tf.float32)
            self.rnn_outputs = tf.add(self.rnn_outputs[0], self.rnn_outputs[1])
        with tf.variable_scope("bi-lstm-r"):
            # Relation
            _fw_cell_2 = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            fw_cell_2 = tf.nn.rnn_cell.DropoutWrapper(_fw_cell_2, self.rnn_dropout_keep_prob)
            _bw_cell_2 = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer())
            bw_cell_2 = tf.nn.rnn_cell.DropoutWrapper(_bw_cell_2, self.rnn_dropout_keep_prob)
            self.rnn_rel_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell_2,
                                                                      cell_bw=bw_cell_2,
                                                                      inputs=self.embedded_r,
                                                                      sequence_length=self._length(self.input_r),
                                                                      dtype=tf.float32)
            self.rnn_rel_outputs = tf.add(self.rnn_rel_outputs[0], self.rnn_rel_outputs[1])

        # Attention of the question words
        with tf.variable_scope('attention'):
            if use_attention:
                self.attn, self.alphas = attention(self.rnn_outputs)

        # Pooling
        with tf.variable_scope('pooing'):
            # Relation
            print(self.rnn_rel_outputs.shape)
            self.rel_representation = tf.layers.max_pooling1d(inputs=self.rnn_rel_outputs,
                                                              pool_size=int(self.rnn_rel_outputs.shape[1]),
                                                              strides=int(self.rnn_rel_outputs.shape[2]))
            print(self.rel_representation)
            self.rel_representation = tf.layers.flatten(self.rel_representation)
            # Question
            print(self.rnn_outputs.shape)
            self.ques_representation = tf.layers.max_pooling1d(inputs=self.rnn_outputs,
                                                              pool_size=int(self.rnn_outputs.shape[1]),
                                                              strides=int(self.rnn_outputs.shape[2]))
            print(self.ques_representation)
            self.ques_representation = tf.layers.flatten(self.ques_representation)

        # Dropout | ps: now the vectors of question and relation should have the same dimension.
        with tf.variable_scope('dropout'):
            if use_attention:
                self.q_drop = tf.nn.dropout(self.attn, self.dropout_keep_prob)
            else:
                self.q_drop = tf.nn.dropout(self.ques_representation, self.dropout_keep_prob)
            self.r_drop = tf.nn.dropout(self.rel_representation, self.dropout_keep_prob)

        # output method 1: Cosine Similarity
        # with tf.variable_scope('cosine'):
        #     self.q_norm = tf.sqrt(tf.reduce_sum(tf.square(self.q_drop), axis=1))
        #     self.r_norm = tf.sqrt(tf.reduce_sum(tf.square(self.r_drop), axis=1))
        #     self.prods = tf.reduce_sum(tf.multiply(self.q_drop, self.r_drop), axis=1)
        #     print(self.prods)
        #     print(self.q_norm)
        #     self.sims = self.prods / (self.q_norm * self.r_norm)
        #     print(self.sims)
        #     self.cos_prob = tf.stack([self.sims, tf.negative(self.sims)], axis=1)
        #     print(self.cos_prob)
            # self.sims = tf.convert_to_tensor(self.sims)
            # self.prob = tf.nn.softmax(self.sims, dim=0)  # shape: (neg_num + 1)*batch
            # self.hit_prob = tf.transpose(selfz.prob[0])
            # self.loss = -tf.reduce_mean(tf.log(self.hit_prob))

        # output method 2: Fully connected layer
        with tf.variable_scope('output'):
            print(self.q_drop)
            print(self.r_drop)
            self.merged = tf.multiply(self.q_drop, self.r_drop)
            self.merged = tf.layers.dense(self.merged, 256, kernel_initializer=initializer(), activation=tf.nn.relu)
            self.merged = tf.nn.dropout(self.merged, self.dropout_keep_prob)
            print(self.merged.shape)
            self.originallogits = tf.layers.dense(self.merged, 2, kernel_initializer=initializer())
            self.logits = tf.nn.softmax(self.originallogits)
            print(self.logits.shape)
            # self.logits = tf.layers.dense(self.h_drop, num_classes, kernel_initializer=initializer())
            # self.predictions = [r[0] for r in self.logits]
            # self.predictions = tf.argmax(self.logits, axis=1, name="predictions")

        # Calculate mean cross-entropy loss | Note this loss function treat each q,r pair (positive/negative) as
        # a separate task. not same as the loss function described in ACL17 (pair-wise ranking)
        with tf.variable_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.originallogits, labels=self.input_y)
            # losses = tf.nn.weighted_cross_entropy_with_logits(targets=self.input_y, logits=self.originallogits, pos_weight=20)
            # losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.originallogits, labels=self.input_y)
            # self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(losses)  # + l2_reg_lambda * self.l2
            # Still need regularization?

        # Accuracy | Calculate accuracy out of this model.
        # with tf.variable_scope("accuracy"):
        #     correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
