import pickle
import random
import yaml

from django.http import HttpResponse
from django.shortcuts import render

from .model import *


# Create your views here.
def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


@run_once
def init_model():

    global dataset

    with open(base + 'data/dataset.p', 'rb') as f:
        dataset = pickle.load(f)

    with open(base + 'model/henryp.init.yaml', 'r') as f:
        init = yaml.load(f)

    henryp = CNN(**init)
    henryp.featurize = lambda dataset: np.stack(dataset.feature.lsa)
    henryp.load_weights(base + 'model/henryp.weight.h5')

    henryp.compile(optimizer='RMSprop',
                   loss='categorical_crossentropy', metrics=['acc'])
    henryp.evaluate(dataset)
    predict = henryp.predict_class(dataset)
    dataset['data', 'predict'] = predict


def nlil(request):
    init_model()

    ind = request.GET.get('ind', random.choice(dataset.data.index))
    Arg1 = ' '.join(dataset.data.Arg1[ind])
    Arg2 = ' '.join(dataset.data.Arg2[ind])
    predict = dataset.data.predict[ind]
    Relation = dataset.data.Relation[ind]
    return render(request, 'vickygod/base.html',
                  {'Arg1': Arg1, 'Arg2': Arg2, 'predict': predict, 'Relation': Relation})
