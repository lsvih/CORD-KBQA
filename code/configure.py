import argparse
import sys

# base_path = "sq/"
base_path = "webqsp/"
use_attention = True

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    # Data loading params
    parser.add_argument("--train_path", default="data/"+base_path+"WebQSP.RE.stype.train",  # SQ.RE.rtype.train | WebQSP.RE.stype.train
                        type=str, help="Path of train data")
    parser.add_argument("--test_path", default="data/"+base_path+"WebQSP.RE.stype.test",
                        type=str, help="Path of test data")
    parser.add_argument("--allow_type", default=False,
                        type=str, help="Using type info of topic entity?")
    parser.add_argument("--allow_para", default=True,
                        type=str, help="Using parallel questions info of relations?")
    parser.add_argument("--max_sentence_length", default=60,   # data_helpers.py: type length:10, pq length: 30
                        type=int, help="Max sentence length in data")
    parser.add_argument("--dev_sample_percentage", default=0.1,
                        type=float, help="Percentage of the training data to use for validation")

    # Model Hyper-parameters
    # Embeddings
    parser.add_argument("--embedding_path", default='./data/glove.6B.300d.txt',
                        type=str, help="Path of pre-trained word embeddings (glove)")
    parser.add_argument("--embedding_dim", default=300,
                        type=int, help="Dimensionality of word embedding (default: 100)")
    parser.add_argument("--emb_dropout_keep_prob", default=1.0,
                        type=float, help="Dropout keep probability of embedding layer (default: 0.7)")
    # AttLSTM
    parser.add_argument("--hidden_size", default=200,
                        type=int, help="Dimensionality of RNN hidden (default: 100)")
    parser.add_argument("--rnn_dropout_keep_prob", default=1.0,
                        type=float, help="Dropout keep probability of RNN (default: 0.7)")
    parser.add_argument("--attention", default=True,    # Not use this, use the global variable for convenience
                        type=bool, help="Allow question attention (or use maxpooling)")

    # Misc
    parser.add_argument("--desc", default="",
                        type=str, help="Description for model")
    parser.add_argument("--dropout_keep_prob", default=1.0,
                        type=float, help="Dropout keep probability of output layer (default: 0.5)")
    parser.add_argument("--l2_reg_lambda", default=1e-5,
                        type=float, help="L2 regularization lambda (default: 1e-5)")

    # Training parameters
    parser.add_argument("--batch_size", default=100,
                        type=int, help="Batch Size (default: 10)")
    parser.add_argument("--num_epochs", default=20,     # 100 for L2 | 20 for SQ/WQSP
                        type=int, help="Number of training epochs (Default: 100)")
    parser.add_argument("--display_every", default=100,   # 5 for L2| 100 for SQ/WQSP|
                        type=int, help="Number of iterations to display training information")
    parser.add_argument("--evaluate_every", default=3000,  # 20 for L2| 3000 for SQ/WQSP|
                        type=int, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument("--num_checkpoints", default=5,
                        type=int, help="Number of checkpoints to store (default: 5)")
    parser.add_argument("--learning_rate", default=1.0,
                        type=float, help="Which learning rate to start with (Default: 1.0)")
    parser.add_argument("--decay_rate", default=0.9,
                        type=float, help="Decay rate for learning rate (Default: 0.9)")

    # Testing parameters | Notice this param do not effect on Training process.
    parser.add_argument("--checkpoint_dir", default="",
                        type=str, help="Checkpoint directory from training run")

    # Misc Parameters
    parser.add_argument("--allow_soft_placement", default=True,
                        type=bool, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", default=False,
                        type=bool, help="Log placement of ops on devices")
    parser.add_argument("--gpu_allow_growth", default=True,
                        type=bool, help="Allow gpu memory growth")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    print("")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg.upper(), getattr(args, arg)))
    print("")

    return args


FLAGS = parse_args()
