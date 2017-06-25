import pandas as pd
import numpy as np
import word2vec
import keras
from keras.models import Model, Sequential
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
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# keras.backend.tensorflow_backend.set_session(session)
base = '/Users/henryyang/Desktop/NLIL/'


def scaled_attention(scale):
    def attention(E):
        W = K.batch_dot(E, E, axes=2)  # dot(E, E.T)
        W /= scale
        W = K.softmax(W)
        return K.batch_dot(W, E, axes=1)
    return attention


class Autoencoder(Model):
    def __init__(self, dim=300, encode_layers=None, decode_layers=None):
        if encode_layers is None:
            encode_layers = [dim // 2, dim // 4]
        if decode_layers is None:
            decode_layers = [dim // 2]
        dim_code = encode_layers[-1]
        self.__dict__.update(locals())

        inputs = Input(shape=(dim, ), name='input')
        enc = [Dense(encode_layers[0], input_shape=(dim,), activation='relu')]
        enc += [Dense(n, activation='relu') for n in encode_layers[1:]]
        self.encoder = encoder = Sequential(enc)

        decoder = [Dense(decode_layers[0], input_shape=(
            dim_code, ), activation='relu')]
        decoder += [Dense(n, activation='relu')
                    for n in decode_layers[1:] + [dim]]
        self.decoder = decoder = Sequential(decoder)

        output = decoder(encoder(inputs))

        super().__init__(inputs=inputs, outputs=output)
        self.init_weights = self.get_weights()

    def reset(self):
        self.set_weights(self.init_weights)

    def fit(self, train, valid=None, **kwargs):
        val = (valid, valid)
        return super().fit(train, train, validation_data=val, **kwargs)


class CNN(Model):
    def __init__(self, wordvec=base + 'data/wordvec.txt', window=1,
                 maxlen=100, filter=250, pe=False,
                 drop_emb=.5, drop_conv=.5, drop_sa=.25, drop_res=0, drop_dense=0,
                 feature=0, n_layers=1, sa_scale=0, residue=False, dense_layers=[],
                 pad_position=['pre', 'pre'], n_relation=4, n_clause=2, torel=None):

        if type(dense_layers) is int:
            dense_layers = [dense_layers]
        self.__dict__.update(locals())
        if torel is not None:
            self.n_relation = n_relation = len(torel)

        self.wordvec = wordvec = word2vec.load(wordvec)
        vs, es = wordvec.vectors.shape

        # Input Layers
        inputs = [Input(shape=(maxlen,), name='clause%d' % i)
                  for i in range(n_clause)]

        if self.feature:
            feature = [Input(shape=(feature,), name='feature')]
        else:
            feature = []

        # Hidden Layers
        embed = Embedding(vs, es, input_length=maxlen,
                          weights=[wordvec.vectors], name='embed')

        if sa_scale:
            attn = Lambda(scaled_attention(sa_scale), name='attention')
            drop_sa = Dropout(drop_sa, name='drop_sa')

        if not hasattr(filter, '__getitem__'):
            filter = (filter, ) * n_layers
        if not hasattr(drop_conv, '__getitem__'):
            drop_conv = (drop_conv, ) * n_layers
        if not hasattr(drop_res, '__getitem__'):
            drop_res = (drop_res, ) * n_layers
        if not hasattr(drop_dense, '__getitem__'):
            drop_dense = (drop_dense, ) * len(dense_layers)

        if self.pe:
            dim = es
            pe_i, pe_p = np.meshgrid(range(dim), range(maxlen))
            pe = np.sin(2 * np.pi * pe_p / maxlen * pe_i / dim)
            #pe = np.sin(pe_p/10000**(2*pe_i/dim))
            #pe /= wordvec.vectors.max()*10
            pe_const = K.constant(pe, name='pe')
            pe = Lambda(lambda x: x + pe_const, name='pos_enc')

        if residue:
            reduce = [[Dense(filter[j], name='reduce%d-%d' % (i, j))
                       for j in range(n_layers)] for i in range(n_clause)]
            add = Add(name='add')
            drop_res = [Dropout(drop_res[j], name='drop_res-%d' % j)
                        for j in range(n_layers)]

        conv = [[Conv1D(filter[j], window, activation='relu', name='conv%d-%d' % (i, j))
                 for j in range(n_layers)] for i in range(n_clause)]
        drop_conv = [Dropout(drop_conv[j], name='drop_conv-%d' % j)
                     for j in range(n_layers)]

        pool = MaxPooling1D(maxlen, padding='same', name='pool')
        drop_emb = Dropout(drop_emb, name='drop_emb')
        flatten = Flatten(name='flatten')
        dense_layers = [Dense(n, activation='relu', name='dense%d' % i)
                        for i, n in enumerate(dense_layers)]
        drop_dense = [Dropout(dp, name='drop_dense-%d' % j)
                      for j, dp in enumerate(drop_dense)]
        dense_out = Dense(n_relation, activation='softmax', name='dense_out')
        concat = Concatenate(name='concat')

        # Forward
        h = [embed(clause) for clause in inputs]
        if self.pe:
            h = [pe(hi) for hi in h]
        h = [drop_emb(hi) for hi in h]

        if sa_scale:
            h = [attn(hi) for hi in h]
            h = [drop_sa(hi) for hi in h]

        for j in range(n_layers):
            if residue:
                r = [reduce[i][j](h[i]) for i in range(n_clause)]
                r = [drop_res[j](ri) for ri in r]
            h = [conv[i][j](h[i]) for i in range(n_clause)]
            h = [drop_conv[j](hi) for hi in h]
            if residue:
                h = [add([h[i], r[i]]) for i in range(n_clause)]

        h = [pool(hi) for hi in h]
        h = [flatten(hi) for hi in h]

        h = concat(h + feature)

        for dense, drop in zip(dense_layers, drop_dense):
            h = drop(dense(h))
        output = dense_out(h)

        super().__init__(inputs=inputs + feature, outputs=output)
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

    def evaluate(self, testset, **kwargs):
        inp, out = self._generate_vector(testset)
        if self.feature:
            inp += [self.featurize(testset)]
        return super().evaluate(inp, out, **kwargs)

    def predict_class(self, testset):
        inp, _ = self._generate_vector(testset)
        if self.feature:
            inp += [self.featurize(testset)]
        out = self.predict(inp)

        return pd.Series(((np.arange(self.n_relation) + 1, self.torel[prob.argsort()[::-1]], prob[prob.argsort()[::-1]]) for prob in out),
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
            return inp, None

    def plot(self, filename):
        from IPython.display import Image
        plot_model(self, to_file=filename)
        return Image(filename=filename)
