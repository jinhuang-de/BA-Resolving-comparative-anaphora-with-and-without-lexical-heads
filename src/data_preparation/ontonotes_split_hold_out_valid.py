from ontonotes_utils import *
from loader_ontonotes_pronoun import *
from sklearn.model_selection import KFold
import json
import numpy as np

# Split and save the test set
# Split and save the k fold training and validation data manually,
# so that we make sure that the anaphors in the same file(article/context)
# dont appear in training and validation set at the same time

class AnaphorEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Anaphor):
            return {
                    "left": o.left,
                    "right": o.right,
                    "tokens": o.tokens,
                    "coref_id": o.coref_id,
                    "grammatical_role": o.grammatical_role,
                    "context": o.context,
                    "golds_ids": o.golds_ids,
                    "golds_str": o.golds_str,

                    "potential_antecedents": o.potential_antecedents,
                    "candidates_distance_features_matrix": o.candidates_distance_features_matrix,
                    "candidates_deps_features_matrix": o.candidates_deps_features_matrix,
                    "definiteness": o.definiteness,
                    "candidates_definiteness_features_matrix": o.candidates_definiteness_features_matrix,

                    "file_name": o.file_name,
                    "head_lemma": o.head_lemma,
                    "candidates_string_match_features": o.candidates_string_match_features,
                    "candidates_synonym_features": o.candidates_synonym_features,
                    "candidates_hypernym_features": o.candidates_hypernym_features}

        if isinstance(o, Mention):
            return {
                    "left": o.left,
                    "right": o.right,
                    "tokens": o.tokens,
                    "coref_id": o.coref_id,
                    "grammatical_role": o.grammatical_role,
                    "definiteness": o.definiteness,
                    "head_lemma": o.head_lemma}

        if isinstance(o, np.ndarray):
            return o.tolist()

        return json.JSONEncoder.default(self, o)


def get_corpus_name(i):
    if i == 0:
        return 'anaphor_with'
    elif i == 1:
        return ('anaphor_without')
    elif i == 2:
        return 'total'


def handle_files_overlaps(list_1, list_2):
    list_1 = list(list_1)
    list_2 = list(list_2)

    list_1_file_names = [x_y[0].file_name for x_y in list_1]
    for_list_2__file_in_list_1_to_check = list_1_file_names[-1]

    list_2_idx_to_remove_and_add_to_list_1 = []

    for idx, x_y in enumerate(list_2):
        if x_y[0].file_name == for_list_2__file_in_list_1_to_check:
            list_2_idx_to_remove_and_add_to_list_1.append(idx)

    list_2_elements_to_remove_and_add_to_list_1 = np.array(list_2)[list_2_idx_to_remove_and_add_to_list_1]
    list_2_elements_to_remove_and_add_to_list_1 = list_2_elements_to_remove_and_add_to_list_1.tolist()
    list_1 = list_1 + list_2_elements_to_remove_and_add_to_list_1

    for idx in list_2_idx_to_remove_and_add_to_list_1:
        list_2.pop(idx)


    return list_1, list_2


if __name__ == '__main__':
    c = Corpus()

    X = c.corpus_pronoun
    y = get_all_gold_labels(X)

    all_comparative_contextes = [x for x in c.corpus_total]


    for i in range(10):
        print(X[i].context)
        print(all_comparative_contextes[i].context)

        print('\n')

    for x in X:
        if x.context in all_comparative_contextes:
            print(x.context)


    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, shuffle=False)



    Xy_train = [[X, y] for X, y in zip(X_train, y_train)]
    Xy_val = [[X, y] for X, y in zip(X_val, y_val)]
    Xy_test = [[X, y] for X, y in zip(X_test, y_test)]


    Xy_train, Xy_val = handle_files_overlaps(Xy_train, Xy_val)
    Xy_val, Xy_test = handle_files_overlaps(Xy_val, Xy_test)


    x_train = [x_y[0] for x_y in Xy_train]
    y_train = [x_y[1] for x_y in Xy_train]

    x_val= [x_y[0] for x_y in Xy_val]
    y_val = [x_y[1] for x_y in Xy_val]

    x_test = [x_y[0] for x_y in Xy_test]
    y_test= [x_y[1] for x_y in Xy_test]


    xtrain_ytrain_xval_y_val = [x_train, x_val, x_test, y_train, y_val, y_test]

    for l in xtrain_ytrain_xval_y_val:
        print(len(l))

    # SAVE K FOLDS LIST FOR EVERY CORPUS# without_gold_corefs/
    with open("../k_folds_corpus/ontonotes___holdout_Xtrain_ytrain_Xval_yval_Xtest_ytest.txt", "w") as fp:
        fp.write(json.dumps(xtrain_ytrain_xval_y_val, cls=AnaphorEncoder))

    print('finished!')