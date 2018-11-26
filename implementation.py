import tensorflow as tf
import re

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

HIDDEN_UNITS = 128 #size of input/output hidden layers and lstm size
NUM_CLASSES = 2 #pos or neg
LEARNING_RATE = 0.001

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """

    processed_review = []
    #remove html tags from reviews
    line = re.sub('<[^>]*>','',review).lower()
    #remove none letters but keep the spaces
    line = re.sub('[^a-z\s]','',line)
    line = line.split()
    #remove stop words
    for e in line:
        if(e not in stop_words):
            processed_review.append(e)
    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    #placeholders for input data, labels, dropout
    input_data = tf.placeholder(tf.float32,[None, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE],
        name="input_data")
    labels = tf.placeholder(tf.int32,[None, NUM_CLASSES],name="labels")
    dropout_keep_prob = tf.placeholder_with_default(0.9,shape=(),name="dropout_keep_prob")
    
    #reshape the input data to 2-D for matmal
    input_data_re = tf.reshape(input_data, [-1, EMBEDDING_SIZE])

    #weights & biases for input layer and output layer
    weights = {

        'input_layer': tf.Variable(tf.random_normal([EMBEDDING_SIZE, HIDDEN_UNITS])),

        'output_layer': tf.Variable(tf.random_normal([HIDDEN_UNITS, NUM_CLASSES]))
    }
    biases = {

        'input_layer': tf.Variable(tf.constant(0.1, shape=[HIDDEN_UNITS, ])),

        'output_layer': tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES, ]))
    }

    #input for RNN layer
    input_data_in = tf.matmul(input_data_re, weights['input_layer']) + biases['input_layer']

    input_data_in = tf.nn.dropout(input_data_in, dropout_keep_prob)
    # reshape back to 3-D
    input_data_in = tf.reshape(input_data_in, [-1, MAX_WORDS_IN_REVIEW, HIDDEN_UNITS])

    #RNN layer
    with tf.name_scope("RNN_LAYER"):

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_UNITS, forget_bias = 1.0)
            #apply dropout to reduce overfitting
            drop = tf.contrib.rnn.DropoutWrapper(
                                    cell=lstm_cell,output_keep_prob=dropout_keep_prob)
    #RNN initial state
    init_state = drop.zero_state(BATCH_SIZE, tf.float32)

    with tf.name_scope("RNN_DYNAMIC_CELL"):
        outputs, states = tf.nn.dynamic_rnn(drop,input_data_in,initial_state=init_state,
                                        time_major = False,dtype=tf.float32)


    #unpack the outputs from the RNN and take out the last result
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['output_layer']) + biases['output_layer']


    # apply dropouts to the result
    preds = tf.contrib.layers.dropout(results, dropout_keep_prob)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels),name="loss")
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    #calculate accuracy
    correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name="accuracy")

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
