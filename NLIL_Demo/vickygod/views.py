import pickle
import random
import re
import yaml

from django.http import HttpResponse
from django.shortcuts import render

from sklearn.model_selection import train_test_split

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
    _, dataset = train_test_split(dataset, test_size=0.25, random_state=42)


def nlil(request):
    init_model()

    ind = int(request.GET.get('ind', random.choice(dataset.data.index)))
    title = dataset.raw.title[ind]
    content = dataset.raw.content[ind]
    predict = np.dstack(dataset.data.predict[ind])[0]
    Relation = dataset.data.Relation[ind]
    content = re.sub(r'(%s)' % Relation, r'<b style="background-color: yellow;">\1</b>', content)

    return render(request, 'vickygod/base.html',
                  {'ind': ind, 'title': title, 'content': content, 'predict': predict, 'Relation': Relation})
