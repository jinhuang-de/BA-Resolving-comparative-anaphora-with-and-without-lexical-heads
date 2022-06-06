# corpus_names = ['anaphor_with', 'anaphor_without', 'total']
from util import *
kfolds = load_corpus_list('total')
k_folds_X_train = kfolds[0]
k_folds_X_val = kfolds[1]
k_folds_X_test = kfolds[2]

for x in range(len(k_folds_X_test)):
    to_check = k_folds_X_test[x]
    all_others = []
    for fold_i, fold in enumerate(k_folds_X_test):
        if fold_i != x:
            all_others = all_others + fold

    to_check_ids = [(test['id'], test['file_name']) for test in to_check]
    others_ids = [(test['id'], test['file_name']) for test in all_others]

    overlap = 0
    for v in to_check_ids:
        if v in others_ids:
            overlap += 1
    print(overlap)
