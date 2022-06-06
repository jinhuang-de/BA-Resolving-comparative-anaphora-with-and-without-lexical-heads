import pandas as pd
import xml.etree.ElementTree as ET
import pandas
import ast
import re
import random
from collections import OrderedDict
from loader_ontonotes_pronoun import *
# from sentence_transformers import SentenceTransformer
# from torch import nn

class statistics():
    def __init__(self, preprocessed_corpus_path):
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.bert_model = SentenceTransformer("whaleloops/phrase-bert")

        self.preprocessed_corpus_path = preprocessed_corpus_path
        self.corpus_path = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/corpus/Dataset_comparative_anaphora_resolution/'
        self.onto_notes_corpus_path = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/corpus/ontonotes-5.0-conll-2012/'
        self.no_head_list = ["other", "others", "one", "ones", "another"]
        self.df = pandas.read_csv(preprocessed_corpus_path, sep='\t', index_col=[0]) # get rid of the 'Unnamed' caused by index
        self.df_with = pandas.read_csv(self.corpus_path + 'preprocessed/anno_with_lexical_info.csv', sep='\t', index_col=[0])
        self.df_without = pandas.read_csv(self.corpus_path + 'preprocessed/anno_without_lexical_info.csv', sep='\t', index_col=[0])
        self.df_onto_notes = pandas.read_csv(self.onto_notes_corpus_path + 'preprocessed/ontonotes_retrieval.csv', sep='\t', index_col=[0])

    def span_to_MinMax_span(self, span):
        '''Turn s string span into a min max span
        @param span:
        str: a span str from the doc e.g. 'word_5..word_7'
        @return:
        list: a min max span e.g. [5, 7]
        '''
        span = re.findall(r'\d+', span)
        # Because span: in AR & VZ: 'word_5..word_5', but in  IS: 'word_5'
        if len(span) < 2:
            MinMax_span = [int(span[0]), int(span[0])]
        else:
            MinMax_span = [int(span[0]), int(span[1])]
        return MinMax_span

    def get_markable_through_id(self, searched_id, f_name):
        '''Find the markable object of a xml tree.
        @param searched_id:
        str: the markable id
        @param f_name:
        str: file name
        @return:
        the markable object--- child.attrib in ET.parse(path).getroot()
        '''
        markables_tree = ET.parse(self.corpus_path+"modified/markables_/" + f_name)
        markables_root = markables_tree.getroot()
        for child in markables_root:
            markable = child.attrib
            if searched_id == markable['id']:
                return markable

    def split_corpus_base_on_lexical_information(self):
        # ist keine Noun/Propernoun oder auf der liste
        df_without = 0
        for index, row in self.df.iterrows():
            anaphor_dict = ast.literal_eval(row['anaphor'])
            markable_id = list(anaphor_dict.keys())[0]
            head_str = anaphor_dict[markable_id]['head']['surface_str']
            head_pos = anaphor_dict[markable_id]['head']['pos']
            # if the head is in the list
            if head_str in self.no_head_list:
                try:
                    if df_without == 0:
                        df_without = pd.DataFrame(row).T
                except ValueError:
                    df_without = df_without.append(pd.DataFrame(row).T)
                # remove it from self.df to build df_with
                self.df.drop(index, inplace=True)
            # if the head is noun or proper noun
            elif head_pos not in ['NOUN', 'PROPN', 'NUM']:
                print(head_pos)
                try:
                    if df_without == 0:
                        df_without = pd.DataFrame(row).T
                except ValueError:
                    df_without = df_without.append(pd.DataFrame(row).T)
                # remove it from self.df to build df_with
                self.df.drop(index, inplace=True)

        df_without.reset_index(drop=True, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        df_without.to_csv(self.corpus_path+'/preprocessed/anno_without_lexical_info.csv', sep='\t')
        self.df.to_csv(self.corpus_path+'/preprocessed/anno_with_lexical_info.csv', sep='\t')

    def find_anaphors_with_more_than_one_antecedents(self): # the 14 annas
        for index, row in self.df.iterrows():
            anaphor_dict = ast.literal_eval(row['anaphor'])
            gold_dict = ast.literal_eval(row['gold_antecedents'])
            context_dict = ast.literal_eval(row['context'])
            file_name = row['file_name']
            ana_markable_id = list(anaphor_dict.keys())[0]

            anaphor_span = anaphor_dict[ana_markable_id]['span']

            if len(gold_dict.keys())>1:
                i = 1
                for key in list(gold_dict.keys()):
                    s = 'gold_antecedent {}.: '.format(i)
                    i +=1

    def antecedent_in_same_and_last_n_sents(self, anaphor_span, gold_span, f_name):
        ### to retrieve the idxs of the context sentences and the anaphor sentence
        sents_tree = ET.parse(self.corpus_path + "original/sentences/" + f_name)
        anaphor_MinMax_span = self.span_to_MinMax_span(anaphor_span)
        gold_MinMax_span = self.span_to_MinMax_span(gold_span)
        sents_root = sents_tree.getroot()
        sents_spans = [self.span_to_MinMax_span(child.attrib['span']) for child in sents_root]  # [[1,19], [20,40], ...]

        anaphor_sent_idx = None
        gold_sent_idx = None
        # Find the anaphor sent span: anaphor_sent_idx
        for sent_span in sents_spans:
            # if the anaphor is in this sentence
            if (sent_span[0] <= anaphor_MinMax_span[0]) and (sent_span[1] >= anaphor_MinMax_span[1]):
                # Xth_sent = sents_idxs.index(sent_idx)  # (start wiz 0)in Xth sentence appears the anaphor(index of sents_idxs: sents_idx[0])
                anaphor_sent_idx = sents_spans.index(sent_span)

        for sent_span in sents_spans:
            if sent_span[0] <= gold_MinMax_span[0] and sent_span[1] >= gold_MinMax_span[1]:
                gold_sent_idx = sents_spans.index(sent_span)

        # handle gold span which is longer then a sentence
        if gold_sent_idx == None:
            for sent_span in sents_spans:
                # if the gold span end is farer than sent span, then the gold span in the anaphor sent span
                if gold_MinMax_span[0] <= sent_span[1] and gold_MinMax_span[1] >= sent_span[1]:
                    gold_sent_idx = sents_spans.index(sent_span)
        n_before_anaphor_ana_sent = anaphor_sent_idx - gold_sent_idx
        return n_before_anaphor_ana_sent

    def antecedent_in_same_and_last_n_sents(self, anaphor_span, gold_span, f_name):
        ### to retrieve the idxs of the context sentences and the anaphor sentence
        sents_tree = ET.parse(self.corpus_path + "original/sentences/" + f_name)
        anaphor_MinMax_span = self.span_to_MinMax_span(anaphor_span)
        gold_MinMax_span = self.span_to_MinMax_span(gold_span)
        sents_root = sents_tree.getroot()
        sents_spans = [self.span_to_MinMax_span(child.attrib['span']) for child in sents_root]  # [[1,19], [20,40], ...]

        pritn(sents_spans)

        anaphor_sent_idx = None
        gold_sent_idx = None
        # Find the anaphor sent span: anaphor_sent_idx
        # for sent_span in sents_spans:
            # if the anaphor is in this sentence
        #     if (sent_span[0] <= anaphor_MinMax_span[0]) and (sent_span[1] >= anaphor_MinMax_span[1]):
                # Xth_sent = sents_idxs.index(sent_idx)  # (start wiz 0)in Xth sentence appears the anaphor(index of sents_idxs: sents_idx[0])
        #         anaphor_sent_idx = sents_spans.index(sent_span)

        for sent_span in sents_spans:
            if sent_span[0] <= gold_MinMax_span[0] and sent_span[1] >= gold_MinMax_span[1]:
                gold_sent_idx = sents_spans.index(sent_span)

        # handle gold span which is longer then a sentence
        if gold_sent_idx == None:
            for sent_span in sents_spans:
                # if the gold span end is farer than sent span, then the gold span in the anaphor sent span
                if gold_MinMax_span[0] <= sent_span[1] and gold_MinMax_span[1] >= sent_span[1]:
                    gold_sent_idx = sents_spans.index(sent_span)
        n_before_anaphor_ana_sent = anaphor_sent_idx - gold_sent_idx
        return n_before_anaphor_ana_sent

    def calculate_ratio_of_antecedents_posistions(self, df):
        # {n sentences before anaphor sentence: count of how many annotation match this window size}
        n_to_count = {}
        df_len = len(df)
        for index, row in df.iterrows():
            anaphor_dict = ast.literal_eval(row['anaphor'])
            gold_dict = ast.literal_eval(row['gold_antecedents'])
            file_name = row['file_name']
            ana_markable_id = list(anaphor_dict.keys())[0]
            # TODO: now we only take care of the first gold ant, we need to consider 2 if we use 2
            gold_markable_id = list(gold_dict.keys())[0]
            anaphor_span = anaphor_dict[ana_markable_id]['span']
            gold_span = gold_dict[gold_markable_id]['span']

            n_before = self.antecedent_in_same_and_last_n_sents(anaphor_span, gold_span, file_name)
            if n_to_count.__contains__(n_before):
                n_to_count[n_before] += 1
            else:
                n_to_count[n_before] = 1

        ratio = {}
        n_to_count = OrderedDict(sorted(n_to_count.items()))
        key_list = list(n_to_count.keys())
        for key in key_list:
            count = 0
            for i in range(key_list.index(key) + 1):
                k = key_list[i]
                count += n_to_count[k]
            ratio[key] = count/df_len

    def onto_notes_calculate_ratio_of_antecedents_posistions(self, df):
        # {n sentences before anaphor sentence: count of how many annotation match this window size}
        n_to_count = {}
        df_len = len(df)
        for index, row in df.iterrows():
            anaphor_dict = ast.literal_eval(row['anaphor'])
            gold_list = ast.literal_eval(row['gold_antecedents'])
            context_list = ast.literal_eval(row['context'])['whole_context']['context_sent_list']

            # TODO: now we only take care of the first gold ant, we need to consider 2 if we use 2
            sents_spans = None
            anaphor_span = anaphor_dict['span_idx_in_context']
            if len(context_list) == 1:
                sent_1_span = [0, len(context_list[0]) + 1]
                sents_spans = [sent_1_span]

            elif len(context_list) == 2:
                sent_1_span = [0, len(context_list[0]) + 1]
                sent_2_span = [len(context_list[0]) , len(context_list[0]) + len(context_list[1]) + 1]
                sents_spans = [sent_1_span, sent_2_span]

            elif len(context_list) == 3:
                sent_1_span = [0, len(context_list[0]) + 1]

                sent_2_span = [len(context_list[0]) , len(context_list[0]) + len(context_list[1]) + 1]
                sent_3_span = [len(context_list[0]) + len(context_list[1]) , len(context_list[0]) + len(context_list[1]) + len(context_list[2]) + 1]
                sents_spans = [sent_1_span, sent_2_span, sent_3_span]

            gold_spans = [gold['span_idx_in_context'] for gold in gold_list]
            gold_span = sorted(gold_spans, key=lambda x: x[1])[-1]

            n_before = None

            for sent_id, sent_span in enumerate(sents_spans):
                if gold_span[0] >= sent_span[0] and gold_span[1] <= sent_span[1]:
                    n_before = sent_id

            if n_before == None:
                print('ana: ', anaphor_dict['surface_str'], anaphor_dict['span_idx_in_context'])
                print('gold', gold_span)

                print(sents_spans)
                print(context_list)
                print('\n')

            if n_to_count.__contains__(n_before):
                n_to_count[n_before] += 1
            else:
                n_to_count[n_before] = 1

        ratio = {}
        n_to_count = OrderedDict(sorted(n_to_count.items()))
        key_list = list(n_to_count.keys())
        for key in key_list:
            count = 0
            for i in range(key_list.index(key) + 1):
                k = key_list[i]
                count += n_to_count[k]
            ratio[key] = count/df_len
        print(n_to_count)


    def onto_average_context_len(self, df):
        # {n sentences before anaphor sentence: count of how many annotation match this window size}
        context_len = 0

        for index, row in df.iterrows():
            context_list = ast.literal_eval(row['context'])['whole_context']['preprocessed_surface_str']
            context_len += len(context_list)
        print(context_len/len(df))

    def frequency_antecedent_as_subject(self, df):
        antecedent_is_subject_count = 0
        for index, row in df.iterrows():
            gold_dict = ast.literal_eval(row['gold_antecedents'])
            if isinstance(gold_dict, list):
                for gold in gold_dict:
                    gold_head_dep = gold['head']['dep']
                    if gold_head_dep == 'nsubj':
                        antecedent_is_subject_count += 1
            else:
                gold_markable_id = list(gold_dict.keys())[0]
                gold_head_dep = gold_dict[gold_markable_id]['head']['dep']
                if gold_head_dep == 'nsubj':
                    antecedent_is_subject_count += 1

        print('antecedent_is_subject_count: ', antecedent_is_subject_count)
        print('ratio: ', antecedent_is_subject_count/len(df))

    def frequency_antecedent_types(self, df):
        definite = ["the", "all", "both", "either", "neither", "no", "none"]
        indefinite = ["a", "an", "each", "every", "some", "any," "few", "several", "many", "much", "little", "most", "more", "fewer", "less"]
        demonstrative = ['this', 'these', 'that', 'those']

        correct_definite_count = 0
        correct_indefinite_count = 0
        correct_no_articel = 0
        correct_demonstrative_count = 0
        correct_proper_name_count = 0
        correct_pronominal_count = 0
        correct_possesive = 0
        correct_total_count = 0

        potential_definite_count = 0
        potential_indefinite_count = 0
        potential_no_articel = 0
        potential_demonstrative_count = 0
        potential_proper_name_count = 0
        potential_pronominal_count = 0
        potential_possesive = 0
        potential_total_count = 0

        # check what type the gold antecedents are
        for index, row in df.iterrows():
            gold_dict = ast.literal_eval(row['gold_antecedents'])
            gold_markable_id = list(gold_dict.keys())[0]
            gold_pos = gold_dict[gold_markable_id]['head']['pos']
            gold_dep = gold_dict[gold_markable_id]['head']['dep']
            gold_surface = gold_dict[gold_markable_id]['surface_str']
            gold_head_surface = gold_dict[gold_markable_id]['head']['surface_str']

            gold_head_children = gold_dict[gold_markable_id]['head']['children']
            gold_head_children = re.findall(r'\w+', gold_head_children) + [gold_head_surface]

            # head_idx = gold_surface.index(gold_head_surface)
            defi = list(set(gold_surface).intersection(set(definite)))
            indefi = list(set(gold_surface).intersection(set(indefinite)))
            demon = list(set(gold_surface).intersection(set(demonstrative)))

            if 'PROPN' == gold_pos:
                correct_proper_name_count += 1
            elif 'poss' == gold_dep or '\'s' in gold_surface:
                correct_possesive += 1
            elif 'PRON' in gold_pos:
                correct_pronominal_count +=1

            # elif len(defi) > 0 and gold_surface.index(defi[0]) < head_idx:
            elif len(defi) > 0 and any(x in gold_head_children for x in defi):
                correct_definite_count += 1
            elif len(indefi) > 0 and any(x in gold_head_children for x in indefi):
                correct_indefinite_count += 1
            elif len(demon) > 0 and any(x in gold_head_children for x in demon):
                correct_demonstrative_count += 1
            else:
                correct_no_articel += 1
            correct_total_count += 1

            # check what type the coreferences of gold antecedents are
            if str(row['coref_chain']) != 'nan':
                coref_dict = ast.literal_eval(row['coref_chain'])
                coref_markable_ids = list(coref_dict.keys())
                for coref_id in coref_markable_ids:
                    coref_pos = coref_dict[coref_id]['head']['pos']
                    coref_dep = coref_dict[coref_id]['head']['dep']
                    coref_surface = coref_dict[coref_id]['surface_str']
                    coref_head_surface = coref_dict[coref_id]['head']['surface_str']

                    coref_head_children = coref_dict[coref_id]['head']['children']
                    coref_head_children = re.findall(r'\w+', coref_head_children) + [coref_head_surface]

                    # head_idx = coref_surface.index(coref_head_surface)
                    defi = list(set(coref_surface).intersection(set(definite)))
                    indefi = list(set(coref_surface).intersection(set(indefinite)))
                    demon = list(set(coref_surface).intersection(set(demonstrative)))

                    if coref_id != gold_markable_id:
                        if 'PROPN' == coref_pos:
                            correct_proper_name_count += 1
                        elif 'poss' == coref_dep or '\'s' in coref_surface:
                            correct_possesive += 1
                        elif 'PRON' == coref_pos:
                            correct_pronominal_count += 1
                        elif len(defi) > 0 and any(x in coref_head_children for x in defi):
                            correct_definite_count += 1
                        elif len(indefi) > 0 and any(x in coref_head_children for x in indefi):
                            correct_indefinite_count += 1
                        elif len(demon) > 0 and any(x in coref_head_children for x in demon):
                            correct_demonstrative_count += 1
                        else:
                            correct_no_articel += 1
                        correct_total_count += 1

            potential_dict = ast.literal_eval(row['potential_antecedent'])
            potential_markable_ids = list(potential_dict.keys())
            if len(potential_markable_ids) > 1:
                for potential_id in potential_markable_ids:
                    potential_pos = potential_dict[potential_id]['head']['pos']
                    potential_dep = potential_dict[potential_id]['head']['dep']
                    potential_surface = potential_dict[potential_id]['surface_str']
                    potential_head_surface = potential_dict[potential_id]['head']['surface_str']

                    potential_head_children = potential_dict[potential_id]['head']['children']
                    potential_head_children = re.findall(r'\w+', potential_head_children) + [potential_head_surface]

                    # head_idx = potential_surface.index(potential_head_surface)
                    defi = list(set(potential_surface).intersection(set(definite)))
                    indefi = list(set(potential_surface).intersection(set(indefinite)))
                    demon = list(set(potential_surface).intersection(set(demonstrative)))

                    if 'PROPN' == potential_pos:
                        potential_proper_name_count += 1
                    elif 'poss' == potential_dep or '\'s' in potential_surface:
                        potential_possesive += 1
                    elif 'PRON' == potential_pos:
                        potential_pronominal_count += 1
                    elif len(defi) > 0 and any(x in potential_head_children for x in defi):
                        potential_definite_count += 1
                    elif len(indefi) > 0 and any(x in potential_head_children for x in indefi):
                        potential_indefinite_count += 1
                    elif len(demon) > 0 and any(x in potential_head_children for x in demon):
                        potential_demonstrative_count += 1
                    else:
                        potential_no_articel += 1
                    potential_total_count += 1

        print('correct_definite_count: ', correct_definite_count/correct_total_count)
        print('correct_indefinite_count: ', correct_indefinite_count/correct_total_count)
        print('correct_demonstrative_count: ', correct_demonstrative_count/ correct_total_count)
        print('correct_proper_name_count: ', correct_proper_name_count/ correct_total_count)
        print('correct_pronominal_count: ', correct_pronominal_count/ correct_total_count)
        print('correct_possesive: ', correct_possesive/ correct_total_count)
        print('ccorrect_no_articel: ', correct_no_articel / correct_total_count, '\n')

        print('potential_definite_count: ', potential_definite_count / potential_total_count)
        print('potential_indefinite_count: ', potential_indefinite_count / potential_total_count)
        print('potential_demonstrative_count: ', potential_demonstrative_count / potential_total_count)
        print('potential_proper_name_count: ', potential_proper_name_count / potential_total_count)
        print('potential_pronominal_count: ', potential_pronominal_count / potential_total_count)
        print('potential_possesive: ', potential_possesive / potential_total_count)
        print('potential_no_articel: ', potential_no_articel / potential_total_count, '\n')

    def random_and_recency_base_lines(self, df):
        random_correct = 0
        recency_correct = 0

        random_correct_coref = 0
        recency_correct_coref = 0

        df_len = len(df)

        # RANDOM
        # for every anaphor
        for index, row in df.iterrows():
            gold_markable_ids = list(ast.literal_eval(row['gold_antecedents']).keys())
            gold_antecedents_list = gold_markable_ids
            correct_antecedents_list = gold_markable_ids  # if the anaphor's antecedents has only gold antecedent(s) but no coreferences
            # if the anaphor's antecedents has coreferences
            if str(row['coref_chain']) != 'nan':
                coref_dict = ast.literal_eval(row['coref_chain'])
                coref_markable_ids = list(coref_dict.keys())
                correct_antecedents_list = list(set(coref_markable_ids + gold_antecedents_list))

            potential_dict = ast.literal_eval(row['potential_antecedent'])
            # seeds = [90, 11, 982, 3201, 123]
            seeds = [1, 2, 3, 4, 5]
            for s in seeds:
                random.seed(s)
                # pick a random potential ants----------------------------------------
                potential_ids = list(potential_dict.keys())
                random_idx = random.randint(0, len(potential_ids) - 1)
                random.shuffle(potential_ids)
                random_potential_markable_id = potential_ids[random_idx]
                potential = potential_dict[random_potential_markable_id]

                # find the coref chain of the picked random potential ant--------------
                random_coref = potential['coref_chain']
                if random_coref!= None:
                    random_coref_ids = set(list(random_coref.keys()) + [random_potential_markable_id])
                else:
                    random_coref_ids = [random_potential_markable_id]

                ### if any coref of the picked random_potential_markable_id is gold ant:
                for random_id in random_coref_ids:
                    if random_id == gold_antecedents_list[0]:
                        random_correct += 1
                        break

                # if any coref of the picked random_potential_markable_id in correct_antecedents_list
                for random_id in random_coref_ids:
                    if random_id in correct_antecedents_list:
                        random_correct_coref += 1
                        break
            # RECENCY---------------------------------------------------------V
            # TODO: längste wählen
            nearest_potential_markables = list(sorted(potential_dict.items(), key=lambda item:item[1]['span'][-4:], reverse=True))

            nearst_span = self.span_to_MinMax_span(nearest_potential_markables[0][1]['span'])
            nearest_pid = nearest_potential_markables[0]

            nearest_coref_ids = list(nearest_pid)

            for potential_markable in nearest_potential_markables:
                pid = potential_markable[0]
                # find the longest from the nearest one
                # if end of the span is the same
                if self.span_to_MinMax_span(potential_markable[1]['span'])[1] == nearst_span[1]:
                    # if the beginning of the span is in front of the old nearest pid
                    if self.span_to_MinMax_span(potential_markable[1]['span'])[0] < nearst_span[0]:
                        # update the new nearest pid
                        nearest_pid = potential_markable[0]
                        # if thid pid has coref chain, update the nearest coref ids
                        if list(potential_markable[1]['coref_chain']) != None and len(potential_markable[1]['coref_chain'].keys())!=0:
                            nearest_coref_ids = list(potential_markable[1]['coref_chain'].keys())
                            nearest_coref_ids.append(pid)
                            nearest_coref_ids = list(set(nearest_coref_ids))

            for nearest_pid in nearest_coref_ids:
                ### if nearest_potential_markable_id in correct_antecedents_list:
                if nearest_pid in gold_antecedents_list:
                    recency_correct += 1
                    break
            for nearest_pid in nearest_coref_ids:
                if nearest_pid in correct_antecedents_list:
                    recency_correct_coref += 1
                    break

        print('ramdom_gold_id: ', random_correct/(df_len*5))
        print('random_correct_coref: ', random_correct_coref / (df_len*5))
        print('recency_gold_id: ', recency_correct / df_len)
        print('recency_correct_coref: ', recency_correct_coref / df_len, '\n')

    def new_random_and_recency_base_lines(self, df):
        random_correct = 0
        recency_correct = 0

        random_correct_coref = 0
        recency_correct_coref = 0

        df_len = len(df)

        # RANDOM
        for index, row in df.iterrows():

            gold_markable_ids = list(ast.literal_eval(row['gold_antecedents']).keys())
            gold_antecedents_list = gold_markable_ids
            # coref_set = list(ast.literal_eval(row['gold_antecedents'])[gold_antecedents_list[0]]['coref_set'])

            correct_antecedents_list = gold_markable_ids  # if the anaphor's antecedents has only gold antecedent(s) but no coreferences
            # if the anaphor's antecedents has coreferences
            if str(row['coref_chain']) != 'nan':
                coref_dict = ast.literal_eval(row['coref_chain'])
                coref_markable_ids = list(coref_dict.keys())
                correct_antecedents_list = list(set(coref_markable_ids + gold_antecedents_list))

            potential_dict = ast.literal_eval(row['potential_antecedent'])
            # seeds = [90, 11, 982, 3201, 123]
            seeds = [1, 2, 3, 4, 5]
            for s in seeds:
                random.seed(s)
                # pick a random potential ants----------------------------------------
                potential_ids = list(potential_dict.keys())
                random_idx = random.randint(0, len(potential_ids) - 1)
                random.shuffle(potential_ids)
                random_potential_markable_id = potential_ids[random_idx]
                potential = potential_dict[random_potential_markable_id]

                # find the coref chain of the picked random potential ant--------------
                random_coref = potential['coref_chain']
                if random_coref!= None:
                    random_coref_ids = set(list(random_coref.keys()) + [random_potential_markable_id])
                else:
                    random_coref_ids = [random_potential_markable_id]

                ### if any coref of the picked random_potential_markable_id is gold ant:
                for random_id in random_coref_ids:
                    if random_id == gold_antecedents_list[0]:
                        random_correct += 1
                        break

                # if any coref of the picked random_potential_markable_id in correct_antecedents_list
                for random_id in random_coref_ids:
                    if random_id in correct_antecedents_list:
                        random_correct_coref += 1
                        break

            # RECENCY---------------------------------------------------------V
            # TODO: längste wählen
            nearest_potential_markables = list(sorted(potential_dict.items(), key=lambda item:item[1]['span'][-4:], reverse=True))

            nearst_span = self.span_to_MinMax_span(nearest_potential_markables[0][1]['span'])
            nearest_pid = nearest_potential_markables[0]

            nearest_coref_ids = list(nearest_pid)

            for potential_markable in nearest_potential_markables:
                pid = potential_markable[0]
                # find the longest from the nearest one
                # if end of the span is the same
                if self.span_to_MinMax_span(potential_markable[1]['span'])[1] == nearst_span[1]:
                    # if the beginning of the span is in front of the old nearest pid
                    if self.span_to_MinMax_span(potential_markable[1]['span'])[0] < nearst_span[0]:
                        # update the new nearest pid
                        nearest_pid = potential_markable[0]
                        # if thid pid has coref chain, update the nearest coref ids
                        if list(potential_markable[1]['coref_chain']) != None and len(potential_markable[1]['coref_chain'].keys())!=0:
                            nearest_coref_ids = list(potential_markable[1]['coref_chain'].keys())
                            nearest_coref_ids.append(pid)
                            nearest_coref_ids = list(set(nearest_coref_ids))

            for nearest_pid in nearest_coref_ids:
                ### if nearest_potential_markable_id in correct_antecedents_list:
                if nearest_pid in gold_antecedents_list:
                    recency_correct += 1
                    break
            for nearest_pid in nearest_coref_ids:
                if nearest_pid in correct_antecedents_list:
                    recency_correct_coref += 1
                    break

        print('ramdom_gold_id: ', random_correct/(df_len*5))
        print('random_correct_coref: ', random_correct_coref / (df_len*5))
        print('recency_gold_id: ', recency_correct / df_len)
        print('recency_correct_coref: ', recency_correct_coref / df_len, '\n')

    def recency_baseline_evaluattion(self, df):
        eva_file = open('../evaluation/recency_baseline_withLex_evaluation.txt', 'a')
        recency_correct = 0
        recency_correct_coref = 0

        df_len = len(df)

        # RANDOM
        for index, row in df.iterrows():
            anaphor_dict = ast.literal_eval(row['anaphor'])
            anaphor_dict = anaphor_dict[list(anaphor_dict.keys())[0]]
            gold_dict = ast.literal_eval(row['gold_antecedents'])
            gold_dict = gold_dict[list(gold_dict.keys())[0]]
            context = ast.literal_eval(row['context'])['whole_context']['preprocessed_surface_str']

            eva_file.write('anaphor: ' + str(anaphor_dict['surface_str']) + '\n')
            eva_file.write('gold: ' + str(gold_dict['surface_str']) + '\n')
            # PREDICTION: find the index of the max value of sigmoid?

            gold_markable_ids = list(ast.literal_eval(row['gold_antecedents']).keys())
            gold_antecedents_list = gold_markable_ids
            # coref_set = list(ast.literal_eval(row['gold_antecedents'])[gold_antecedents_list[0]]['coref_set'])

            correct_antecedents_list = gold_markable_ids  # if the anaphor's antecedents has only gold antecedent(s) but no coreferences
            # if the anaphor's antecedents has coreferences
            if str(row['coref_chain']) != 'nan':
                coref_dict = ast.literal_eval(row['coref_chain'])
                coref_markable_ids = list(coref_dict.keys())
                correct_antecedents_list = list(set(coref_markable_ids + gold_antecedents_list))

            potential_dict = ast.literal_eval(row['potential_antecedent'])


            # RECENCY---------------------------------------------------------
            # TODO: längste wählen
            nearest_potential_markables = list(sorted(potential_dict.items(), key=lambda item:item[1]['span'][-4:], reverse=True))

            nearst_span = self.span_to_MinMax_span(nearest_potential_markables[0][1]['span'])
            nearest_pid = nearest_potential_markables[0]

            nearest_coref_ids = list(nearest_pid)


            for potential_markable in nearest_potential_markables:
                pid = potential_markable[0]
                # find the longest from the nearest one
                # if end of the span is the same
                if self.span_to_MinMax_span(potential_markable[1]['span'])[1] == nearst_span[1]:
                    # if the beginning of the span is in front of the old nearest pid
                    if self.span_to_MinMax_span(potential_markable[1]['span'])[0] < nearst_span[0]:
                        # update the new nearest pid
                        nearest_pid = potential_markable[0]
                        # if thid pid has coref chain, update the nearest coref ids
                        if list(potential_markable[1]['coref_chain']) != None and len(potential_markable[1]['coref_chain'].keys())!=0:
                            nearest_coref_ids = list(potential_markable[1]['coref_chain'].keys())
                            nearest_coref_ids.append(pid)
                            nearest_coref_ids = list(set(nearest_coref_ids))

            if isinstance(nearest_pid[1], dict):
                eva_file.write('selected cand: ' + ' '.join(nearest_pid[1]['surface_str']) + '\n')
            else:
                eva_file.write('selected cand: ' + nearest_pid[1] + '\n')
            eva_file.write('context: ' + ' '.join(context) + '\n')

            # for nearest_pid in nearest_coref_ids:
                ### if nearest_potential_markable_id in correct_antecedents_list:
            if nearest_coref_ids[0] in gold_antecedents_list:
                recency_correct += 1
                # break
            if nearest_coref_ids[0] in correct_antecedents_list:
                recency_correct_coref += 1
                eva_file.write('right\n')
            else:
                eva_file.write('wrong\n')

            eva_file.write('\n')
        print('recency_gold_id: ', recency_correct / df_len)
        print('recency_correct_coref: ', recency_correct_coref / df_len, '\n')

    def ontonotes_random_and_recency_base_lines(self, df):
        random_correct_coref = 0
        recency_correct_coref = 0

        df_len = len(df)

        # RANDOM
        for index, row in df.iterrows():
            gold_coref_id = ast.literal_eval(row['anaphor'])['coref_id']

            candidates = ast.literal_eval(row['potential_antecedent'])

            seeds = [1, 2, 3, 4, 5]
            for s in seeds:
                random.seed(s)
                # pick a random potential ants----------------------------------------
                random_idx = random.randint(0, len(candidates) - 1)
                random.shuffle(candidates)
                selected_candidate = candidates[random_idx]

                # if any coref of the picked random_potential_markable_id in correct_antecedents_list
                if selected_candidate['coref_id'] == gold_coref_id:
                    random_correct_coref += 1

            # RECENCY---------------------------------------------------------V
            # TODO: längste wählen
            nearest_potential_markables = sorted(candidates, key=lambda item:item['span_idx_in_context'][1], reverse=True)

            nearst_cand_coref_id = nearest_potential_markables[0]['coref_id']

            if nearst_cand_coref_id == gold_coref_id:
                recency_correct_coref += 1

        print('random_correct_coref: ', random_correct_coref / (df_len*5))
        print('recency_correct_coref: ', recency_correct_coref / df_len, '\n')

    def average_potential_antecedent_count(self, df):
        print('number of samples: ', df.index)
        # len_is_one = 0
        total_potential_count = 0
        total_potential_count_count_only_one_coref = 0

        coref_ids = None

        df_len = len(df)
        for index, row in df.iterrows():
            correct_antecedents_list = None
            gold_dict = ast.literal_eval(row['gold_antecedents'])

            gold_markable_ids = list(gold_dict.keys())
            gold_antecedents_list = gold_markable_ids

            # if the anaphor's antecedents has coreferences
            if str(row['coref_chain']) != 'nan':
                coref_dict = ast.literal_eval(row['coref_chain'])
                coref_markable_ids = list(coref_dict.keys())
                correct_antecedents_list = list(set(coref_markable_ids + gold_antecedents_list))
            else:
                correct_antecedents_list = gold_antecedents_list

            potential_dict = ast.literal_eval(row['potential_antecedent'])
            potential_ids = list(potential_dict.keys())
            f_name = row['file_name']

            first_correct_antecedent = False
            for p_id in potential_ids:
                # find corefs of potential antecedent
                potential = potential_dict[p_id]
                potential_coref = potential['coref_chain']
                potential_coref_ids = None

                if potential.__contains__('coref_chain') and potential_coref != None:
                    potential_coref_ids = list(potential_coref.keys())
                    potential_coref_ids.append(p_id)
                    potential_coref_ids = list(set(potential_coref_ids))
                else:
                    potential_coref_ids = list(p_id)

                coref_set = []
                for c_id in potential_coref_ids:
                    c_markable = self.get_markable_through_id(c_id, f_name)

                    if c_markable != None: #WHY???????
                        coref_set.append(c_markable['coref_set'])

                total_potential_count_count_only_one_coref += len(set(coref_set))
            total_potential_count += len(potential_ids)
        print('average_potential_antecedent_count: ', total_potential_count/df_len)
        print('total_potential_count_count_only_one_coref: ', total_potential_count_count_only_one_coref/df_len)

    def ontonotes_average_potential_antecedent_count(self, df):
        print('number of samples: ', len(df))
        total_potential_count = 0
        total_potential_count_count_only_one_coref = 0

        coref_ids = None

        df_len = len(df)
        for index, row in df.iterrows():
            potential_list = ast.literal_eval(row['potential_antecedent'])

            total_potential_count += len(potential_list)
        print('average_potential_antecedent_count: ', total_potential_count / df_len)


    def sort_csv_per_file_names(self):
        df = pandas.read_csv(self.preprocessed_corpus_path, sep='\t', index_col=[0])

        # sort data frame
        df.sort_values(["file_name"],
                            axis=0,
                            ascending=[False],
                            inplace=True)
        df.to_csv(self.preprocessed_corpus_path, sep='\t')

    def gold_ana_dis(self):
        corpus = Corpus()
        all_anaphors_with = corpus.corpus_with
        all_anaphors_without = corpus.corpus_without
        all_anaphors_total = corpus.corpus_total
        all_corpus = [all_anaphors_with, all_anaphors_without, all_anaphors_total]

        for corpus in all_corpus:
            sum_dis = 0
            sum_gold_in_cand = 0
            for ana in corpus:
                potentials_spans = [[p.left, p.right] for p in ana.potential_antecedents]
                potentials_spans = sorted(potentials_spans, key=lambda x: x[1])
                # print(potentials_spans, '\n')
                gold_span = [ana.gold.left, ana.gold.right]
                if gold_span in potentials_spans:
                    dis = len(potentials_spans) - potentials_spans.index(gold_span)
                    sum_dis += dis
                    sum_gold_in_cand += 1

            print(sum_dis/sum_gold_in_cand)

    def ontonotes_gold_ana_dis(self):
        corpus = Corpus()
        all_anaphors_with = corpus.corpus_with
        all_anaphors_without = corpus.corpus_without
        all_anaphors_total = corpus.corpus_total
        all_corpus = [all_anaphors_with, all_anaphors_without, all_anaphors_total]

        for corpus in all_corpus:
            sum_dis = 0
            sum_gold_in_cand = 0
            for ana in corpus:
                potentials_spans = [[p.left, p.right] for p in ana.potential_antecedents]
                potentials_spans = sorted(potentials_spans, key=lambda x: x[1])
                # print(potentials_spans, '\n')
                gold_span = [ana.gold.left, ana.gold.right]
                if gold_span in potentials_spans:
                    dis = len(potentials_spans) - potentials_spans.index(gold_span)
                    sum_dis += dis
                    sum_gold_in_cand += 1

            print(sum_dis/sum_gold_in_cand)

    def ontonotes_gold_ana_dis(self):
        corpus = Corpus().corpus_pronoun

        sum_dis = 0
        sum_gold_in_cand = 0

        for ana in corpus:
            potentials_spans = [[p.left, p.right] for p in ana.potential_antecedents]
            potentials_spans = sorted(potentials_spans, key=lambda x: x[1])
            golds = ana.gold
            golds_spans = [[gold.left, gold.right] for gold in golds]
            # choose the nearest in the coref chian (in the context) as gold
            gold_span = [sorted(golds_spans, key=lambda x: x[1])][-1]
            if gold_span in potentials_spans:
                dis = len(potentials_spans) - potentials_spans.index(gold_span)
                sum_dis += dis
                sum_gold_in_cand += 1

        print(sum_dis/sum_gold_in_cand)


    def main(self):
        # self.sort_csv_per_file_names()
        # self.split_corpus_base_on_lexical_information()
        df_list = [self.df_with, self.df_without]
        # self.gold_ana_dis()
        for df in df_list:
            # self.onto_average_context_len(df)
            # self.frequency_antecedent_as_subject(df)
            # self.recency_baseline_evaluattion(df)
            self.antecedent_in_same_and_last_n_sents(df)
            # self.average_potential_antecedent_count(df)
            # self.frequency_antecedent_types(df)
        # df = self.df_onto_notes
        # self.ontonotes_gold_ana_dis()
        # self.onto_notes_calculate_ratio_of_antecedents_posistions(df)
        # self.onto_average_context_len(df)
        # self.ontonotes_average_potential_antecedent_count(df)
        # self.calculate_ratio_of_antecedents_posistions(df)
        # self.antecedent_in_same_and_last_n_sents(df)
        # self.frequency_antecedent_as_subject(df)
        # self.ontonotes_average_potential_antecedent_count(df)
        # self.frequency_antecedent_types(df)
        # self.ontonotes_random_and_recency_base_lines(self.df_onto_notes)

if __name__ == '__main__':
    preprocessed_corpus_path = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/corpus/Dataset_comparative_anaphora_resolution/preprocessed/annotation_retrieval.csv'
    sta = statistics(preprocessed_corpus_path)
    sta.main()