import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

hist_names = dict(
    acc = ('acc', 'train'),
    val_acc = ('acc', 'valid'), 
    loss = ('loss', 'train'), 
    val_loss = ('loss', 'valid')
)
def record(hist):
    history = pd.DataFrame(hist.history, index=pd.Index(hist.epoch, name='epoch')).rename(columns=hist_names)
    history.columns = pd.MultiIndex.from_tuples(history.columns)
    
    print('Max Valid Acc Epoch')
    print(history[history.acc.valid == history.acc.valid.max()])
    history.acc.plot(title='Accuracy')
    history.loss.plot(title='Loss')
