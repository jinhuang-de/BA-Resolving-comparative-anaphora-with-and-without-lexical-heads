import re
import torch
from sklearn.model_selection import train_test_split
import json
from nltk.corpus import wordnet

def span_to_span_idx(span):
    '''Turn s string span into a span idx of the whole context
    @param span:
    str: a span str from the doc e.g. 'word_5..word_7'
    @return:
    list: a min max span e.g. [4, 6]
    '''
    span = re.findall(r'\d+', span)
    # Because span: in AR & VZ: 'word_5..word_5', but in  IS: 'word_5'
    if len(span) < 2:
        span_idx = [int(span[0])-1, int(span[0])]
    else:
        span_idx = [int(span[0])-1, int(span[1])]
    return span_idx

def span_idx_to_str_slice(left:int, right:int):
    '''Convert span idx to a slice in string
    e.g. (1, 2) => '[1:2]'
    @param left: start index
    @param right: end index
    @return: a slice in string
    '''
    return '[' + str(left) + ':' + str(right) + ']'

def to_cuda(x):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def batch(iterable, batch_size=1):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx:min(ndx + batch_size, length)]

def split_train_set(X):
    '''Splot the dataset into training, validation and test sets
    @param X:
    @param y:
    @return:
    '''
    # SORTED
    X.sort(key=lambda x: x.file_name, reverse=False)
    y = get_all_gold_labels(X)
    # STAY SORTED
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=None, shuffle=False)
    # print([x.file_name for x in X_train])
    # print(X_train, y_train, X_test,y_test)
    return X_train, y_train, X_test,y_test

def assign_label(sigmoid_score):
    ''' All the score <= 0.5 will be assigned to 0 and > 0.5 will be assigned to 1.
    @param sigmoid_score: Tensor
    @return: int: 1 or 0
    '''
    score = float(sigmoid_score)
    if score > 0.5:
        return 1
    else:
        return 0

def get_all_gold_labels(corpus):
    '''Get all gold labels of the whole corpus(all the anaphors)
    @param corpus: a list of all the anaphor objects
    @return: a list of gold labels of the whole corpus(all the anaphors)
    '''
    corpus_labels = []
    for ana in corpus:
        # find out the gold label and the coreferences of this anaphor
        ana_gold_and_corefs = ana.golds_ids # AS CORRECT: GOLD LABEL AND ITS COREFERENCES
        # if the anaphor has corefs

        potentials_selfs_and_corefs = [p.coref_id for p in ana.potential_antecedents] # list of lists

        # PRODUCE BINARY LABEL for every anaphor(the whole corpus)
        potentials_labels_per_ana = []
        for p_self_and_corefs in potentials_selfs_and_corefs:
            # if any p_corefs same as any ana_corefs_ids
            if any(x in p_self_and_corefs for x in ana_gold_and_corefs):
                potentials_labels_per_ana.append(1)
            else:
                potentials_labels_per_ana.append(0)
        corpus_labels.append(potentials_labels_per_ana)

    return corpus_labels


def my_3d_concat(t1, t2):
    ''' Concatenate 2 3d tensors(beause tensor cant do it due to size problem)
    @param l1: torch.tensor
    @param l2: torch.tensor
    @return: a concatenation of the 2 tensors
    '''
    all = []
    for a1, b1 in zip(t1, t2):
        # print(a1, b1)
        all.append(torch.cat([a1, b1], 1))
    return torch.stack(all)

def get_batch_distance_features_matrixs(docs):
    ''' get a batch of the distance features matrix of all the candidates of each anaphor
    @param docs: a list of anaphor objects
    @return: a list of a batch the distance features matrix of all the candidates of each anaphor
    '''
    return [ana['candidates_distance_features_matrix'] for ana in docs] # [ana[potentials]]

def get_batch_grammar_features_matrixs(docs):
    ''' get a batch of the grammatical roles features matrix of all the candidates of each anaphor
    @param docs: a list of anaphor objects
    @return: a list of a batch the grammatical roles features matrix of all the candidates of each anaphor
    '''

    return [ana['candidates_deps_features_matrix'] for ana in docs] # [ana[potentials]]

def get_batch_definiteness_features_matrixs(docs):
    ''' get a batch of the grammatical roles features matrix of all the candidates of each anaphor
    @param docs: a list of anaphor objects
    @return: a list of a batch the grammatical roles features matrix of all the candidates of each anaphor
    '''

    return [ana['candidates_definiteness_features_matrix'] for ana in docs] # [ana[potentials]]

def get_batch_match_features_scores(docs):
    return [ana['candidates_string_match_features'] for ana in docs] # [ana[potentials]]

def get_batch_synonym_features_scores(docs):
    return [ana['candidates_synonym_features'] for ana in docs] # [ana[potentials]]

def get_batch_hypernym_features_scores(docs):
    return [ana['candidates_hypernym_features'] for ana in docs] # [ana[potentials]]

def remove_duplicantes_in_ontonotes():
    remove = 0

    ontonotes_df =  pd.read_csv('/home/students/huang/Documents/corpus/ontonotes-5.0-conll-2012/preprocessed/ontonotes_retrieval.csv', sep='\t', index_col=[0])
    print('original ontonotes: ', len(ontonotes_df))

    main_df = pd.read_csv('/home/students/huang/Documents/corpus/Dataset_comparative_anaphora_resolution/preprocessed/annotation_retrieval.csv', sep='\t', index_col=[0])

    main_corpus_files_names = [row['file_name'].split('.')[0] for index, row in main_df.iterrows()]

    for index, row in ontonotes_df.iterrows():
        file_name = row['file_name'].split('.')[0]
        if file_name in main_corpus_files_names:
            remove += 1
            ontonotes_df.drop(index, inplace=True)


    print('main_corpus: ', len(main_df))
    print('ontonotes: ', len(ontonotes_df))
    print('removed: ', remove)

    save_to_path = '../../corpus/ontonotes-5.0-conll-2012/preprocessed/ontonotes_retrieval_no_dups.csv'
    ontonotes_df.to_csv(save_to_path, sep='\t')

def load_corpus_list():
    with open("../k_folds_corpus/ontonotes/ontonotes___holdout_Xtrain_ytrain_Xval_yval_Xtest_ytest.txt", 'r') as f:
        train_val_test_corpus = json.load(f)

    return train_val_test_corpus


def get_semantic_feature_definiteness(children_of_head):
    definite = ["the", "all", "both", "either", "neither", "no", "none"]
    indefinite = ["a", "an", "each", "every", "some", "any," "few", "several", "many", "much", "little", "most", "more",
                  "fewer", "less"]
    demonstrative = ['this', 'these', 'that', 'those']

    if any(x in definite for x in children_of_head):
        return 'definite'
    elif any(x in indefinite for x in children_of_head):
        return 'indefinite'
    elif any(x in demonstrative for x in children_of_head):
        return 'demonstrative'


def get_word_synonyms_from_sent(word, sent):
    word_synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemma_names():
            if lemma in sent and lemma != word:
                word_synonyms.append(lemma)
    return word_synonyms