from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# hyperparameter
batch_size = 32
window_size = 5
vocab_size = None
hidden_size = 10
emb_dim = 50
learning_rate = 0.8
epoch_size = 1
    

def load(filepath, window_size, vocab_size=None):

    words = []
    with open(filepath, 'r', encoding='utf8') as file:
        words = word_tokenize(file.readline())    

    x_train, y_train = [], []
    for i in range(len(words) - window_size + 1):
        x_train.append(words[i: i + window_size - 1])
        y_train.append(words[i +  window_size - 1])
    
    vocab = [word[0] for word in Counter(words).most_common(vocab_size)]
    word2id = { vocab[i]: i for i in range(len(vocab)) }
    
    return np.array(x_train), np.array(y_train)[:,None], np.array(vocab), word2id
            
def convert_to_id(x_train, y_train, vocab):
    
    word_to_id = {}
    for i, vocab in enumerate(vocab):
        word_to_id[vocab] = i
        
    for i in range(len(x_train)):
        x_train[i] = [word_to_id[word] for word in x_train[i]]
        y_train[i] = word_to_id[y_train[i][0]]
        
    return x_train.astype(int), y_train.astype(int)


def next_batch(x_train, y_train, batch_size):
    
    num_batch = len(x_train) // batch_size
    for n in range(num_batch):        
        offset = n * batch_size
        x_batch = x_train[offset: offset + batch_size]
        y_batch = y_train[offset: offset + batch_size]
        
        yield x_batch, y_batch
        
def train(model, epoch_size):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True)) as sess:
        initializer = tf.global_variables_initializer()
        initializer.run()
        step = 0
        for epoch in range(epoch_size):
            for x_batch, y_batch in next_batch(x_train, y_train, batch_size):
                feed_dict = {model.input_words: x_batch, model.output_word: y_batch}
                fetches = [model.loss, model.optimizer]
                Loss, _ = sess.run(fetches, feed_dict)
                if step % 100 == 0:
                    print('Step {}, Loss: {}'.format(step, Loss))
                step += 1

    print('Training Done.')
    word_embedding = model.C.eval()
    return model, word_embedding
	
def cosine_similarity(wordvec1, wordvec2):
    return np.dot(wordvec1, wordvec2) / (np.linalg.norm(wordvec1) * np.linalg.norm(wordvec2))

def distance(wordvec1, wordvec2):
    return (np.linalg.norm(wordvec1 - wordvec2))
    
def most_similar(word_embedding, word, n=10):
    
    target = word2id[word]
    top10 = [('', 1) for i in range(n)]
    for i in range(len(word_embedding)):
        if i == target:
            continue
        sim = distance(word_embedding[target], word_embedding[i])
        for j in range(n):
            if sim <= top10[j][1]:
                top10[j+1:] = top10[j:-1]
                top10[j] = (vocab[i], sim)
                break        
    pprint(top10)
	
def plot_word_embedding(word_embedding, vocab):
	pca = PCA(n_components=2)
	wordemb_2D = pca.fit_transform(word_embedding)
	npoint = 10
	x_coords = wordemb_2D[:npoint, 0]
	y_coords = wordemb_2D[:npoint, 1]
	
	plt.scatter(x_coords, y_coords, c='b')
	for label, x, y in zip(vocab[:npoint], x_coords, y_coords):
			plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
	plt.show()
	


class NNLM:
    def __init__(self, batch_size, window_size, vocab_size, emb_dim, hidden_size, learning_rate):
        ##### Model Parameter Definition #####

        # Input && Output
        self.input_words = tf.placeholder(dtype=tf.int32, shape=(batch_size, window_size-1))
        self.output_word = tf.placeholder(dtype=tf.int32, shape=(batch_size, 1))

        # Word Features
        self.C = tf.Variable(tf.truncated_normal(shape=(vocab_size, emb_dim), mean=-1, stddev=-1), name='word_embedding')

        # Hidden Layer Weight && Bias
        self.H = tf.Variable(tf.random_normal(shape=(hidden_size, (window_size - 1 ) * emb_dim)))
        self.d = tf.Variable(tf.random_normal(shape=(hidden_size, )))

        # Hidden-to-Output Weight && Bias
        self.U = tf.Variable(tf.random_normal(shape=(vocab_size, hidden_size)))
        self.b = tf.Variable(tf.random_normal(shape=(vocab_size, )))

        # Projection-to-Output Weight
        self.W = tf.Variable(tf.random_normal(shape=(vocab_size, (window_size - 1) * emb_dim)))

        ##### End #####


        ###### y = b + Wx + Utanh(d + Hx) #####

        # x = (C(w(t-1)), C(w(t-2), ..., C(w(t-n+1))), n == window_size
        with tf.name_scope('Projection_Layer'):
            x  = tf.nn.embedding_lookup(self.C, self.input_words) # (batch_size, window_size-1, emb_dim)
            x  = tf.reshape(x, shape=(batch_size, (window_size - 1) * emb_dim)) # (batch_size, (window_size-1 * emb_dim))
            
        with tf.name_scope('Hidden_Layer'):
            Hx = tf.matmul(x, tf.transpose(self.H)) # (batch_size, hidden_size)
            o  = tf.add(self.d, Hx) # (batch_size, hidden_size)
            a  = tf.nn.tanh(o)  # (batch_size, hidden_size)
            
        with tf.name_scope('Output_Layer'):
            Ua = tf.matmul(a, tf.transpose(self.U)) # (batch_size, vocab_size)
            Wx = tf.matmul(x, tf.transpose(self.W)) # (batch_size, vocab_size)
            y  = tf.nn.softmax(tf.clip_by_value(tf.add(self.b, tf.add(Wx, Ua)), 0.0, 10)) # (batch_size, vocab_size)
            

        with tf.name_scope('Loss'):
            onehot_tgt = tf.one_hot(tf.squeeze(self.output_word), vocab_size)  # (batch_size, vocab_size)
            self.loss = -1 * tf.reduce_mean(tf.reduce_sum(tf.log(y) * onehot_tgt, 1)) # 乘 -1 -> maximize loss
            
        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss) 

        ##### End #####






if __name__ == '__main__':


    filepath = 'corpus/text8-less.txt'

    x_raw, y_raw, vocab, word2id = load(filepath, window_size, vocab_size)
    vocab_size = len(vocab)
    print('vocab_size: {}'.format(vocab_size))

    x_train, y_train = convert_to_id(x_raw, y_raw, vocab)
    print('Length: {}'.format(len(x_train)))
    print('Number of batch: {}'.format(len(x_train) / batch_size))

    model = NNLM(batch_size, window_size, vocab_size, emb_dim, hidden_size, learning_rate)
    model, word_embedding = train(model, epoch_size)

    np.save('wordvector/vector-{}.npy'.format(emb_dim), word_embedding)



