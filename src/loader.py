from typing import *
import pandas
from util import *
import ast
import numpy as np
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet


class Mention:
    def __init__(self, mention_dict, coref_dict=None, is_coref=False):
        self.id = list(mention_dict.keys())[0]
        mention_dict = mention_dict[self.id]
        self.left = span_to_span_idx(mention_dict['span'])[0]
        self.right = span_to_span_idx(mention_dict['span'])[1]
        self.tokens = mention_dict['lemma']
        self.corefs = None
        self.head_str = mention_dict['head']['surface_str']
        self.head_lemma = mention_dict['head']['lemma']

        self.grammatical_role = mention_dict['head']['dep']
        children_of_head = mention_dict['head']['children']
        children_of_head = re.findall(r'\w+', children_of_head)
        self.definiteness = get_semantic_feature_definiteness(children_of_head)

        # only anaphor, potential and gold antecedent have, but thier coferences not.
        if is_coref == False: # if we not handling corefs of mentions
            # a coref list with all mention information except thier coreferences
            self.corefs = []
            if coref_dict != None: # if there is corefs for this mention
                for k, v in coref_dict.keys():
                    coref_mention_dict = {k: v}
                    self.corefs.append(Mention(coref_mention_dict, is_coref=True))
            else: # # if there is no corefs for this mention
                self.corefs = None
        else: # if we r handling corefs of mentions
            self.corefs = None

class Anaphor():
    def __init__(self, file_name, ana_dict, coref_dict, gold_dict, potential_dict, context_dict, is_coref=False):
        self.id = list(ana_dict.keys())[0]
        ana_dict = ana_dict[self.id]
        # FOR THE CORRECT INDEX SLICING
        self.left = span_to_span_idx(ana_dict['span'])[0]
        self.right = span_to_span_idx(ana_dict['span'])[1]
        self.tokens = ana_dict['lemma']
        self.grammatical_role = ana_dict['head']['dep']
        children_of_head = ana_dict['head']['children']
        children_of_head = re.findall(r'\w+', children_of_head)
        self.definiteness = get_semantic_feature_definiteness(children_of_head)
        self.corefs_ids = None
        self.head_str = ana_dict['head']['surface_str']
        self.head_lemma = ana_dict['head']['lemma']

        # only Anaphor Object has it---
        self.context = context_dict['whole_context']['preprocessed_surface_str']
        self.first_index_of_context = span_to_span_idx(context_dict['whole_context']['span'])[0]
        self.gold_id = Mention(gold_dict).id
        self.gold_str = gold_dict[self.gold_id]['lemma']
        self.potential_antecedents = [Mention({k: v}) for k, v in potential_dict.items()]
        self.gold = Mention(gold_dict)

        ### FEATURES:
        # A:
        # DISTANCE
        self.candidates_distance_features_matrix = self.get_candidates_distance_features_matrix() # [num_candidates, 9]
        # GRAMMATICAL
        self.candidates_deps_features_matrix = self.get_ana_concat_candidates_grammar_fetures_matrix() # [num_candidates, 10]
        # DEFINITENESS
        self.candidates_definiteness_features_matrix = self.get_ana_concat_candidates_definiteness_fetures_matrix() # [num_candidates, 3]

        #B:
        # append each of then to pair
        # MATCH
        self.candidates_string_match_features = self.get_candidates_string_match_scroes() # len of candidates
        # SYNONYM
        self.candidates_synonym_features = self.get_candidates_synonym_features() # len of candidates
        # HYPERNYM
        self.candidates_hypernym_features = self.get_candidates_hypernym_features() # len of candidates

        # only anaphor, potential and gold antecedent have, but thier coferences not.
        if is_coref == False: # if we not handling corefs of mentions
            # a coref list with all mention information except thier coreferences
            self.corefs_ids = []
            if coref_dict != None: # if there is corefs for this mention
                for k, v in coref_dict.items():
                    coref_mention_dict = {k: v}
                    self.corefs_ids.append(Mention(coref_mention_dict, is_coref=True).id)
            else:
                self.corefs_ids = None
        else: # if we r handling corefs of mentions
            self.corefs_ids = None

        # maybe useful
        self.file_name = file_name

    def get_candidates_hypernym_features(self):
        def get_word_hypernym_from_sent(anaphor, candidate):
            '''to check whether the anaphor is a hypernym of the respective candidate
            '''
            try:
                anaphor = wordnet.synsets(anaphor)[0]
            except IndexError:
                return 0
            try:
                candidate = wordnet.synsets(candidate)[0]
                candidate_hypernyms = candidate.hypernyms()
                if anaphor in candidate_hypernyms:
                    return 1
                else:
                    return 0
            except IndexError:
                return 0

        ana_head = self.head_lemma
        candidates_head_strings = [cand.head_lemma for cand in self.potential_antecedents]
        candidates_hypernym_features = []

        for cand_head in candidates_head_strings:
            candidates_hypernym_features.append(get_word_hypernym_from_sent(ana_head, cand_head))

        return candidates_hypernym_features

    def get_candidates_synonym_features(self):
        def get_word_synonyms_feature_from_sent(anaphor, candidate):
            feature_score = 0
            try:
                for synset in wordnet.synsets(candidate):
                    for lemma in synset.lemma_names():
                        if lemma == anaphor and lemma != candidate:
                            feature_score = 1
            except IndexError:
                return 0

            return feature_score

        ana_head = self.head_str
        candidates_head_strings = [cand.head_str for cand in self.potential_antecedents]
        candidates_synonym_features = []

        for cand_head in candidates_head_strings:
            candidates_synonym_features.append(get_word_synonyms_feature_from_sent(ana_head, cand_head))

        return candidates_synonym_features

    def get_candidates_string_match_scroes(self):
        ana_string = self.tokens
        candidates_strings = [cand.tokens for cand in self.potential_antecedents]

        candidates_match_scores = []
        # intersection
        for cand_str in candidates_strings:
            amount_of_intersection = len(set(ana_string) & set(cand_str))
            match_score = amount_of_intersection / len(cand_str)
            candidates_match_scores.append(match_score)
        return candidates_match_scores

    def get_candidates_distance_features_matrix(self):
        # ONE HOT ENCODE DISRANCE FEATURES
        candidates_distances = [i for i in range(len(self.potential_antecedents))]
        candidates_distances.reverse()
        buckets = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 7), (8, 15), (16, 31), (32, 200)]
        buckets_idxs = []  # [1, 8, 6]
        for c in candidates_distances:
            for b_id, bucket in enumerate(buckets):
                if c <= bucket[1] and c >= bucket[0]:
                    buckets_idxs.append(b_id)
                    break

        onehot = np.zeros([len(candidates_distances), len(buckets)])
        for cand_idx, buc_idx in enumerate(buckets_idxs):
            onehot[cand_idx, buc_idx] = 1

        return onehot # for one sample: [num_candidates, 9]

    def get_ana_concat_candidates_grammar_fetures_matrix(self):
        candidates_deps = [cand.grammatical_role for cand in self.potential_antecedents]

        ana_type = self.grammatical_role

        # len = 10, concat_size = 20
        types = ["csubj", "csubjpass", "dative", "dobj", "nsubj", "nsubjpass", "obj", "pobj", "iobj", "oprd"]

        # build anaphor one hot vector metrics
        ana_idx = None
        for b_id, type in enumerate(types):
            if ana_type == type:
                ana_idx = b_id
                break
        ana_onehot = np.zeros([len(candidates_deps), len(types)])
        for cand_idx, _ in enumerate(candidates_deps):
            ana_onehot[cand_idx, ana_idx] = 1

        # build candidates one hot vectors metrix
        types_idxs = []  # [1, 8, 6]
        for c_dep in candidates_deps:
            for b_id, type in enumerate(types):
                if c_dep == type:
                    types_idxs.append(b_id)
                    break

        cands_onehot = np.zeros([len(candidates_deps), len(types)])
        for cand_idx, type_idx in enumerate(types_idxs):
            cands_onehot[cand_idx, type_idx] = 1

        concat_ana_cands_onehot_matrix = np.concatenate([ana_onehot, cands_onehot], 1)
        return concat_ana_cands_onehot_matrix

    def get_ana_concat_candidates_definiteness_fetures_matrix(self):
        candidates_definiteness = [cand.definiteness for cand in self.potential_antecedents]

        ana_type = self.definiteness

        # len = 10, concat_size = 20
        types = ['definite', 'indefinite', 'demonstrative']

        # build anaphor one hot vector metrics
        ana_idx = None
        for b_id, type in enumerate(types):
            if ana_type == type:
                ana_idx = b_id
                break
        ana_onehot = np.zeros([len(candidates_definiteness), len(types)])
        for cand_idx, _ in enumerate(candidates_definiteness):
            ana_onehot[cand_idx, ana_idx] = 1

        # build candidates one hot vectors metrix
        types_idxs = []  # [1, 8, 6]
        for c_dep in candidates_definiteness:
            for b_id, type in enumerate(types):
                if c_dep == type:
                    types_idxs.append(b_id)
                    break

        cands_onehot = np.zeros([len(candidates_definiteness), len(types)])
        for cand_idx, type_idx in enumerate(types_idxs):
            cands_onehot[cand_idx, type_idx] = 1

        concat_ana_cands_onehot_matrix = np.concatenate([ana_onehot, cands_onehot], 1)
        return concat_ana_cands_onehot_matrix


class Corpus():
    def __init__(self):
        self.path_without = '../../corpus/Dataset_comparative_anaphora_resolution/preprocessed/anno_without_lexical_info.csv'
        self.path_with = '../../corpus/Dataset_comparative_anaphora_resolution/preprocessed/anno_with_lexical_info.csv'
        self.path_total = '../../corpus/Dataset_comparative_anaphora_resolution/preprocessed/annotation_retrieval.csv'

        self.df_with = pandas.read_csv(self.path_with, sep='\t', index_col=[0])
        self.df_without = pandas.read_csv(self.path_without, sep='\t', index_col=[0])
        self.df_total = pandas.read_csv(self.path_total, sep='\t', index_col=[0])

        self.corpus_with = self.load_anaphor_samples(self.df_with)
        self.corpus_without = self.load_anaphor_samples(self.df_without)
        self.corpus_total = self.load_anaphor_samples(self.df_total)

        self.vocab = self.get_vocab()

    def get_vocab(self):
        all_tokens = []
        for an in self.corpus_total:
            all_tokens = all_tokens + an.context
        return list(set(all_tokens))

    def load_anaphor_samples(self, df):
        all_anaphor = []

        for index, row in df.iterrows():
            ana_dict = ast.literal_eval(row['anaphor'])
            gold_dict = ast.literal_eval(row['gold_antecedents'])
            potential_dict = ast.literal_eval(row['potential_antecedent'])
            file_name = row['file_name']
            context_dict = ast.literal_eval(row['context'])
            # markable_id = list(ana_dict.keys())[0]
            if str(row['coref_chain']) != 'nan':
                coref_dict = ast.literal_eval(row['coref_chain'])
            else:
                coref_dict = None

            all_anaphor.append(Anaphor(file_name, ana_dict, coref_dict, gold_dict, potential_dict, context_dict))
        return all_anaphor