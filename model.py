import keras
from keras.models import Model
from keras.layers import (
    Input, Dense, Flatten, Concatenate,
    Embedding, Conv1D, MaxPooling1D, Dropout,
    Lambda, Activation
)
import keras.backend as K
from keras.layers.merge import Add, Dot
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
import word2vec
import tensorflow as tf
import pandas as pd
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)

def scaled_attention(scale):
  def attention(E):
    W = K.batch_dot(E, E, axes=2) # dot(E, E.T)
    W /= scale
    W = K.softmax(W)
    return K.batch_dot(W, E, axes=1)
  return attention

class CNN(Model):
  def __init__(self, wordvec='data/wordvec.txt', window=1, 
      maxlen=100, filter=250, drop_emb=.5, drop_conv=.5, drop_sa=.25,
      feature=0, n_layers=1, sa_scale=0, residue=False, 
      pad_position=['pre', 'pre'], n_relation=4, n_clause=2, torel=None):

    self.__dict__.update(locals())
    if torel is not None:
      self.n_relation = n_relation = len(torel)

    wordvec = word2vec.load(wordvec)
    vs, es = wordvec.vectors.shape
    self.wordvec = wordvec

    # Input Layers
    inputs = [Input(shape=(maxlen,), name='clause%d'%i) for i in range(n_clause)]

    if self.feature:
      feature = [Input(shape=(feature,), name='feature')]
    else:
      feature = []

    # Hidden Layers
    embed = Embedding(vs, es, input_length=maxlen, 
        weights=[wordvec.vectors], name='embed')

    if sa_scale:
      attn = Lambda(scaled_attention(sa_scale), name="attention")
      drop_sa = Dropout(drop_sa, name='drop_sa')

    if not hasattr(filter, '__getitem__'):
      filter = (filter, ) * n_layers
    if not hasattr(drop_conv, '__getitem__'):
      drop_conv = (drop_conv, ) * n_layers

    if residue:
      reduce = [Dense(filter[-1], name='reduce%d'%i) for i in range(n_clause)]
      add = Add(name='add')

    conv = [tuple(Conv1D(filter[j], window, activation='relu', name='conv%d-%d'%(i, j)) for j in range(n_layers)) for i in range(n_clause)]
    drop_conv = tuple(Dropout(drop_conv[j], name='drop_conv-%d'%j) for j in range(n_layers))

    pool = MaxPooling1D(maxlen, padding='same', name='pool')
    drop_emb = Dropout(drop_emb, name='drop_emb')
    flatten = Flatten(name='flatten')
    dense = Dense(n_relation, activation='softmax', name='dense')
    concat = Concatenate(name='concat')

    # Forward
    h = [embed(clause) for clause in inputs]
    h = [drop_emb(hi) for hi in h]

    if sa_scale:
      h = [attn(hi) for hi in h]
      h = [drop_sa(hi) for hi in h]

    if residue:
      r = [reduce[i](h[i]) for i in range(n_clause)]

    for j in range(n_layers):
      h = [conv[i][j](h[i]) for i in range(n_clause)]
      h = [drop_conv[j](hi) for hi in h]

    if residue:
      h = [add(h[i], r[i]) for i in range(n_clause)]
    
    h = [pool(hi) for hi in h]
    h = [flatten(hi) for hi in h]

    h = concat(h+feature)

    output = dense(h)

    super().__init__(inputs=inputs+feature, outputs=output)
    self.init_weights = self.get_weights()

  def reset(self):
    self.set_weights(self.init_weights)
  def fit(self, trainset, valset=None, **kwargs):
    inp, out = self._generate_vector(trainset)
    if self.feature:
      inp += [self.featurize(trainset)]
    if valset is not None:
      valin, valout = self._generate_vector(valset)
      if self.feature:
        valin += [self.featurize(valset)]
      val = (valin, valout)
      return super().fit(inp, out, validation_data=val, **kwargs)
    else:
      return super().fit(inp, out, **kwargs)

  def predict_class(self, testset):
    inp = self._generate_vector(testset)
    if self.feature:
      inp += [self.featurize(testset)]
    out = self.predict(inp)
    return pd.Series((self.torel[x] for x in out.argmax(1)), 
                     index=testset.data.index,
                     name='Law')

  def _generate_vector(self, dataset):
    def getid(wlist): return [toid.get(x, unk) for x in wlist]
    toid = self.wordvec.vocab_hash
    vocab_size = len(toid)
    unk = toid['<unk>']

    c1 = dataset.data.Arg1.map(getid).values
    c2 = dataset.data.Arg2.map(getid).values
    c1 = pad_sequences(c1, self.maxlen, padding=self.pad_position[0])
    c2 = pad_sequences(c2, self.maxlen, padding=self.pad_position[1])
    inp = [c1, c2]

    if 'Relation' in dataset.data:
      if self.torel is None:
        self.torel = dataset.data.Relation.unique()
      toid = {r: i for i, r in enumerate(self.torel)}
      out = dataset.data.Relation.map(toid.get).tolist()
      out = to_categorical(out, num_classes=self.n_relation)
      return inp, out

    else:
      return inp
  
  def plot(self, filename):
    from IPython.display import Image
    plot_model(self, to_file=filename)
    return Image(filename=filename)

