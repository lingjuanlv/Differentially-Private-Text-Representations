import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
# tf1.0
if tf.__version__=='2.0.0-beta1':
  tf.compat.v1.flags.DEFINE_float('epsilon', 10, "DP epsilon")
  tf.compat.v1.flags.DEFINE_float('alpha', 5, "alpha")
  tf.compat.v1.flags.DEFINE_string('scheme', 'OME', "OME|SUE|OUE")
  tf.compat.v1.flags.DEFINE_string('dataset', 'yelp', "yelp|amazon|imdb")
  FLAGS = tf.compat.v1.flags.FLAGS
else:
  tf.flags.DEFINE_float('epsilon', 10, "DP epsilon")
  tf.flags.DEFINE_float('alpha', 100, "alpha")
  tf.flags.DEFINE_string('scheme', 'OME', "OME|SUE|OUE")
  tf.flags.DEFINE_string('dataset', 'yelp', "yelp|amazon|imdb")
  FLAGS = tf.flags.FLAGS



plt.style.use('ggplot')
def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

eps=FLAGS.epsilon

def flip(x,p):
  return 1-x if np.random.random() < p else x

def float_to_binary(x, m, n):
    """Convert the float value `x` to a binary string of length `m + n`
    where the first `m` binary digits are the integer part and the last
    'n' binary digits are the fractional part of `x`.
    """
    x_scaled = round(x * 2 ** n)
    return '{:0{}b}'.format(x_scaled, m + n).replace('-','1')

def embedding_perturb_BF(embedding,eps,perturb=0):
  # embedding_max: 0.1409158
  m=6
  n=5
  l=m+n
  print('embedding_max:',np.max(embedding))
  print('embedding_min:',np.min(embedding))

  if FLAGS.scheme=='OME':
    # ME
    embedding_encode=np.zeros([embedding.shape[0],embedding.shape[1]*l])
    for i in range(embedding.shape[0]):
      for j in range(embedding.shape[1]):
        embedding_encode[i][j*l:j*l+l]=list(float_to_binary(float(embedding[i][j]), m, n))
    print(embedding_encode.shape)    
    alpha=FLAGS.alpha
    p1=alpha/(1+alpha)
    p2=1/(1+alpha**3)
    # sensitivity:embedding_encode.shape[1]=l*r,r=embedding_dim=50
    q = 1.0 / (1+alpha*np.math.exp(eps/embedding_encode.shape[1]))
    print('p1:',p1)
    print('p2:',p2)
    print('q:',q)
  if FLAGS.scheme=='SUE':
    l=int(np.max(embedding))
    print('domain_siz',l)
    embedding_encode=np.zeros([embedding.shape[0],embedding.shape[1]*l])
    for i in range(embedding.shape[0]):
        for j in range(embedding.shape[1]):
            embedding_encode[i][j*l+int(embedding[i][j])]=1
    print(embedding_encode.shape) 
    embedding_dim=embedding.shape[1]
    p_SUE=np.math.exp(eps/(2*embedding_dim))/(1+np.math.exp(eps/(2*embedding_dim)))
    q_SUE=1/(1+np.math.exp(eps/(2*embedding_dim)))
    print('p_SUE:',p_SUE)
    print('q_SUE:',q_SUE)
  if FLAGS.scheme=='OUE':
    embedding_encode=np.zeros([embedding.shape[0],embedding.shape[1]*l])
    for i in range(embedding.shape[0]):
        for j in range(embedding.shape[1]):
            embedding_encode[i][j*l+int(embedding[i][j])]=1 
    p_OUE=1/2
    embedding_dim=embedding.shape[1]
    q_OUE=1/(1+np.math.exp(eps/(2*embedding_dim)))
    print('p_OUE:',p_OUE)
    print('q_OUE:',q_OUE)

  embedding_encode_perturb=np.zeros([embedding_encode.shape[0],embedding_encode.shape[1]])
  if perturb==0:
    embedding_encode_perturb=embedding_encode
  else:
    for i in range(embedding_encode.shape[0]):
      for j in range(embedding_encode.shape[1]):
          if embedding_encode[i][j]==1:
            if FLAGS.scheme=='OME':
              if j%2==0:
                embedding_encode_perturb[i][j]=flip(embedding_encode[i][j],1-p1)
              else:
                embedding_encode_perturb[i][j]=flip(embedding_encode[i][j],1-p2)

            if FLAGS.scheme=='SUE':
              embedding_encode_perturb[i][j]=flip(embedding_encode[i][j],1-p_SUE)
            if FLAGS.scheme=='OUE':
              embedding_encode_perturb[i][j]=flip(embedding_encode[i][j],1-p_OUE)
          else:
            if FLAGS.scheme=='OME':
              embedding_encode_perturb[i][j]=flip(embedding_encode[i][j],q)
            if FLAGS.scheme=='SUE':
              embedding_encode_perturb[i][j]=flip(embedding_encode[i][j],q_SUE)
            if FLAGS.scheme=='OUE':  
              embedding_encode_perturb[i][j]=flip(embedding_encode[i][j],q_OUE)
  return embedding_encode_perturb


filepath_dict = {'yelp':   'data/sentiment_analysis/yelp_labelled.txt',
                 'amazon': 'data/sentiment_analysis/amazon_cells_labelled.txt',
                 'imdb':   'data/sentiment_analysis/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)
print(df.iloc[0])


df_yelp = df[df['source'] == FLAGS.dataset]
sentences = df_yelp['sentence'].values
y = df_yelp['label'].values


sentences_train, sentences_test, y_train, y_test = train_test_split(
   sentences, y, test_size=0.2, random_state=1000)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
print("vocab_size:",vocab_size)
print(sentences_train[2])
print(X_train[2])

from keras.preprocessing.sequence import pad_sequences

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print(X_train[0, :])

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

embedding_dim = 50
embedding_matrix = create_embedding_matrix(
    '../../../../../Dropbox/TP_test/glove.6B.50d.txt',
    tokenizer.word_index, embedding_dim)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
nonzero_elements / vocab_size

pretrained_model = Sequential()
pretrained_model.add(layers.Embedding(vocab_size, embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=False))
pretrained_model.add(layers.GlobalMaxPool1D())
pretrained_model.summary()

for layer in pretrained_model.layers:
    layer.trainable=False


model = Sequential()
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 2: train on mapped representation w/o DP, using X_train_embedding
X_train_embedding=pretrained_model.predict(X_train)
X_test_embedding=pretrained_model.predict(X_test)

# 2: train on mapped representation with DP, using X_train_embedding_perturb
X_train_embedding_perturb=embedding_perturb_BF(X_train_embedding,eps,perturb=1)
X_test_embedding_perturb=embedding_perturb_BF(X_test_embedding,eps,perturb=1)

history = model.fit(X_train_embedding_perturb, y_train,
                    epochs=50,
                    verbose=False,
                    validation_data=(X_test_embedding_perturb, y_test),
                    batch_size=32)
loss, accuracy = model.evaluate(X_train_embedding_perturb, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test_embedding_perturb, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
