from collections import Counter, defaultdict

import keras.optimizers as optim
import numpy as np
import pandas as pd
import word2vec
import pickle
from django.http import HttpResponse
from django.shortcuts import render
from scipy.sparse import csr_matrix

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from .ckip import *
from .model import *

# Create your views here.
base = '/Users/henryyang/Desktop/NLIL/'
f_wordvec = base + 'data/wordvec.txt'
f_selected_frame = base + 'data/selected_frames.data.p'


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


@run_once
def init():
    global toid, henryp, selected_frames
    wordvec = word2vec.load(f_wordvec)
    toid = wordvec.vocab_hash
    selected_frames = pickle.load(open(f_selected_frame, 'rb'))
    words = (selected_frames.Arg1 + selected_frames.Arg2).map(getid)

    counter = defaultdict(int)
    for r, wlist in enumerate(words):
        if r >= len(words):
            print(r)
    for c in wlist:
        counter[r, c] += 1

    row, col, data = [list(l) for l in zip(*[(r, c, d)
                                             for (r, c), d in counter.items()])]
    shape = len(words), len(wordvec.vocab)
    counter_mat = csr_matrix((data, (row, col)), shape=shape)
    tfidf = TfidfTransformer().fit_transform(counter_mat)
    tfidf_arr = tfidf.toarray()
    tfidf_table = pd.Series(list(tfidf_arr), index=words.index)
    selected_frames['feature', 'tfidf'] = tfidf_table

    n_svd = 300
    svd = TruncatedSVD(n_svd)
    normalizer = Normalizer()
    lsa = make_pipeline(svd, normalizer)
    lsa_vectors = lsa.fit_transform(tfidf)
    lsa_table = pd.Series(list(lsa_vectors), name=(
        'feature', 'lsa'), index=words.index)
    selected_frames['feature', 'lsa'] = lsa_table

    # trainset, validset = train_test_split(selected_frames, test_size=0.25, random_state=42)
    law_count = pd.Series.from_csv(base + 'data/occurrence.csv')
    relation25 = law_count[:25].index
    selected_frames25 = selected_frames[selected_frames.data.Relation.isin(
        relation25)]
    trainset25, validset25 = train_test_split(
        selected_frames25, test_size=0.25, random_state=42)

    print('# Train set'.ljust(20), '%5d' % len(trainset25))
    print('# Valid set'.ljust(20), '%5d' % len(validset25))
    print('# Law'.ljust(20), '%5d' % len(relation25))

    henryp = model.CNN(maxlen=450, torel=relation,
                       dense_layers=[200, 100], drop_dense=0.3,
                       residue=True, drop_res=0.5,
                       filter=100, drop_emb=0.7, feature=n_svd)
    henryp.featurize = lambda dataset: np.stack(dataset.feature.lsa)
    henryp.fit(trainset25, validset25, epochs=1, batch_size=128)


def getid(wlist):
    unk = toid['<unk>']
    return [toid.get(x, unk) for x in wlist]


def nlil(request):
    init()
    if request.method == 'POST':
        user_input = request.POST['input']
        seg_result = seg(user_input)
        toks = [x[0] for x in seg_result.raw]
        words = getid(toks)

        return HttpResponse(str(words))

    return render(request, 'vickygod/base.html', {})
