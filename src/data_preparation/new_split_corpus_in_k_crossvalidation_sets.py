from util import *
from loader import *
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
            return {"id": o.id,
                    "left": o.left,
                    "right": o.right,
                    "tokens": o.tokens,
                    "grammatical_role": o.grammatical_role,
                    "context": o.context,
                    "first_index_of_context": o.first_index_of_context,
                    "gold": o.gold,
                    "gold_id": o.gold_id,
                    "gold_str": o.gold_str,
                    "potential_antecedents": o.potential_antecedents,
                    "candidates_distance_features_matrix": o.candidates_distance_features_matrix,
                    "candidates_deps_features_matrix": o.candidates_deps_features_matrix,
                    "definiteness": o.definiteness,
                    "candidates_definiteness_features_matrix": o.candidates_definiteness_features_matrix,
                    "corefs_ids": o.corefs_ids,
                    "file_name": o.file_name,
                    "head_str": o.head_str,
                    "candidates_string_match_features": o.candidates_string_match_features,
                    "candidates_synonym_features": o.candidates_synonym_features,
                    "candidates_hypernym_features": o.candidates_hypernym_features}

        if isinstance(o, Mention):
            return {"id": o.id,
                    "left": o.left,
                    "right": o.right,
                    "tokens": o.tokens,
                    "grammatical_role": o.grammatical_role,
                    "corefs": o.corefs,
                    "definiteness": o.definiteness,
                    "head_str": o.head_str}

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

    list_2_elements_to_remove_and_add_to_list_1 = np.array((list_2))[list_2_idx_to_remove_and_add_to_list_1]
    list_2_elements_to_remove_and_add_to_list_1 = list_2_elements_to_remove_and_add_to_list_1.tolist()
    list_1 = list_1 + list_2_elements_to_remove_and_add_to_list_1

    for idx in list_2_idx_to_remove_and_add_to_list_1:
        list_2.pop(idx)

    lists = [list_1, list_2]

    return list_1, list_2


if __name__ == '__main__':
    corpus = Corpus()
    all_anaphors_with = corpus.corpus_with
    all_anaphors_without = corpus.corpus_without
    all_anaphors_total = corpus.corpus_total

    all_corpus = [all_anaphors_with, all_anaphors_without, all_anaphors_total]

    corpus_id = 0
    for corpus in all_corpus:

        k_folds_x_train = []
        k_folds_x_val = []
        k_folds_x_test = []
        k_folds_y_train = []
        k_folds_y_val = []
        k_folds_y_test = []

        corpus_name = get_corpus_name(corpus_id)
        X = corpus
        Y = get_all_gold_labels(X)

        X_y = [(x, y) for x, y in zip(X, Y)]

        five_folds_X_y = np.array_split(X_y, 5) # numpy arrays
        five_folds_X_y = [fold.tolist() for fold in five_folds_X_y]

        corssvalid_sets = []
        for Xy_test in five_folds_X_y:
            Xy_trains = []
            Xy_vals = []
            for Xy_val in five_folds_X_y:
                if Xy_test == Xy_val: continue
                Xy_train = [set for set in five_folds_X_y if set != Xy_test and set != Xy_val]
                Xy_train = sum(Xy_train, [])
                Xy_trains.append(Xy_train)
                Xy_vals.append(Xy_val)
                Xy_test = Xy_test

            corssvalid_sets.append([Xy_trains, Xy_vals, Xy_test])

        # SAVE K FOLDS LIST FOR EVERY CORPUS# without_gold_corefs/
        with open("../k_folds_corpus/{c_name}___crossvalidation_sets.txt".format(
                c_name=corpus_name), "w") as fp:
            fp.write(json.dumps(corssvalid_sets, cls=AnaphorEncoder))

        corpus_id += 1
    print('finished!')

