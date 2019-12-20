from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *
from torchvision.models import resnext50_32x4d

from pathlib import Path
import pandas as pd
import numpy as np
import os

def path2fn(path): return path.name
def top_3_accuracy(preds, targs): return top_k_accuracy(preds, targs, 3)

def name(n=10, print_it=True):
    name = "".join(random.choice(string.ascii_lowercase) for _ in range(n))
    if print_it: print(name)
    return name

def get_datasource(dir_name='22k_2sec_better_centered_mel_db', split_idx=0):
    trn_df = pd.read_csv('data/train.csv')

    trn_paths = list(Path(f'data/img_train_{dir_name}/').iterdir())
    tst_paths = list(Path(f'data/img_test_{dir_name}/').iterdir())

    splits = pd.read_pickle('data/splits.pkl')

    trn_df.fname = trn_df.fname.apply(lambda x: x.split('.')[0] + '.png')
    trn_df.set_index('fname', inplace=True)
    lbl_dict = trn_df.to_dict()['label']

    tfms = [[PILImage.create], [path2fn, lbl_dict.__getitem__, Categorize()]]
    dsrc = DataSource(trn_paths, tfms, splits=splits[split_idx])

    return dsrc

from torchvision.models import resnext50_32x4d

def get_learner(dbch, opt_func = partial(Adam, lr=slice(3e-3), wd=0.01, eps=1e-8),
                config=cnn_config(ps=0.33), arch=resnext50_32x4d, cbs=[]):
    return cnn_learner(
        dbch,
        arch,
        opt_func=opt_func,
        metrics=[accuracy, top_3_accuracy],
        config=config,
    )

def create_submission_and_submit(learner, dbch, sub_name, dir_name, preds=None):
    tst_paths = list(Path(f'data/img_test_{dir_name}/').iterdir())

    if preds is None:
        preds = learner.get_preds(dl=test_dl(dbch, tst_paths))[0]

    predicted_label_idxs = preds.argsort(descending=True)[:, :3]
    fns, predicted_labels = [], []

    for path, idxs in zip(tst_paths, predicted_label_idxs):
        fns.append(f'{path.stem}.wav')
        predicted_labels.append([dbch.vocab[idx] for idx in idxs])

    sub = pd.DataFrame({'fname': fns, 'label': predicted_labels})
    sub.label = sub.label.apply(lambda lst: ' '.join(lst))
    sub.to_csv(f'data/submissions/{sub_name}.csv.zip', compression='zip', index=False)

    os.system(f'kaggle competitions submit -c freesound-audio-tagging -f data/submissions/{sub_name}.csv.zip -m {sub_name}')
