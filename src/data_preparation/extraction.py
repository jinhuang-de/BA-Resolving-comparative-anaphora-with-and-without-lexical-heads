import xml.etree.ElementTree as ET
import re, unicodedata
from os import listdir
import pandas as pd
from tqdm import tqdm
import spacy
from collections import OrderedDict
from spacy.tokens import Doc
from typing import List
import inflect
from nltk.corpus import stopwords


class Retriever:
    '''
    - To extract the data we need.
    - ALL entities contains features {surface_str, pos, span, number}
    main():
    Return df with columns: file_name, anaphor, coref_chain, gold_antecedents, potential_antecedent
    '''

    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.file_names = listdir(self.corpus_path + "original/annotations")
        self.all_markables_of_all_docs = {}
        self.df_annotations = pd.DataFrame(columns=['file_name'])
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.tokenizer = self.custom_spacy_tokenizer

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

    def find_potential_antecedents(self, anaphor_span, f_name, whole_context_idxs):
        '''Find all potential antecedents---
        all markables in the context sentence and the anaphor sentence except anaphor itself (include gold antecedents)
        @param anaphor_span:
        str: a span from the doc e.g. 'word_5..word_7'
        @param f_name:
        str: the file
        @param whole_context_idxs:
        dict: with the the min max span of context sentences and anaphor sentence
        e.g.{'cont_sents_span':, 'ana_sent_span':}
        @return:
        dict: a dictionary {potential1_id:{feture1. feture2, ..}, ..}
        '''
        ### to get all potential atecedents(markables, whose span is in the context sentences and the anaphor sentences, except the anaphor itself)
        potential_antecedents = {}
        anaphor_MinMax_span = self.span_to_MinMax_span(anaphor_span)  # [272, 273]
        whole_context_span = [whole_context_idxs['cont_sents_span'][0], whole_context_idxs['ana_sent_span'][1]]
        markables_tree = ET.parse(self.corpus_path + "modified/markables_/" + f_name)
        markables_root = markables_tree.getroot()
        for child in markables_root:  # each markable of all annotation files
            markable = child.attrib
            markable_span = markable['span']
            markable_MinMax_span = self.span_to_MinMax_span(markable_span)

            ## search all protential anteccedent(all markables in context), except anaphor itself!
            if markable_MinMax_span[0] >= whole_context_span[0] and markable_MinMax_span[1] <= anaphor_MinMax_span[0] and markable_MinMax_span != anaphor_MinMax_span:
                markable_surface_str = self.get_tokenized_surface_strings(f_name, markable_MinMax_span, get_tokenized_strings=True, get_surface_string=False)
                head_infos = self.find_the_head(markable_span, f_name)
                lemma_list, pos_list, number_list = self.get_words_attributes_list(markable_span, f_name)
                dep_list, _ = self.get_dep_list_from_word_file(markable_span, f_name)
                markable_id = markable['id']

                try:
                    coref_set = markable['coref_set']
                except KeyError:
                    coref_set = None
                # coref_chain = self.get_coref_chain(markable_id, f_name, markable_span)
                potential_antecedents[markable_id] = {'span': markable_span, 'surface_str': markable_surface_str,
                                                      'lemma': lemma_list,
                                                      'pos': pos_list, 'number': number_list, 'dep': dep_list,
                                                      'head': head_infos, 'coref_set': coref_set}
        return potential_antecedents

    def get_markable_through_id(self, searched_id, f_name):
        '''Find the markable object of a xml tree.
        @param searched_id:
        str: the markable id
        @param f_name:
        str: file name
        @return:
        the markable object--- child.attrib in ET.parse(path).getroot()
        '''
        markables_tree = ET.parse(self.corpus_path + "modified/markables_/" + f_name)
        markables_root = markables_tree.getroot()
        for child in markables_root:
            markable = child.attrib
            if searched_id == markable['id']:
                return markable

    # TODO: THIS FUNKTIONG IS WRONG!!!!!! => TO DELETE
    '''def get_coref_chain(self, gold_id, f_name, anaphor_span):  # to be continued
        To find all the corefrences of the gold antecedent EXCEPT itself,
        which appeared BEFORE the anaphor
        @param gold_ids_list:
        list: a list of the id of the gold antecedent markables
        @param f_name:
        str: the file name
        @param anaphor_span:
        str: a span from the doc e.g. 'word_5..word_7'
        @return:
        dict: a dictionary of the coreferences of the gold antecedents
        e.g. {gold_id_1: {coreferences}, gold_id_2: {coreferences}}

        anaphor_MinMax_span = self.span_to_MinMax_span(anaphor_span)
        markables_tree = ET.parse(self.corpus_path+"modified/markables_/" + f_name)
        markables_root = markables_tree.getroot()
        coref_chains = {}

        # get the gold markable through id
        gold_markable = self.get_markable_through_id(gold_id, f_name)
        # if the antecedent has coreferences
        if gold_markable.__contains__('coref_set'):
            coref_set = gold_markable['coref_set']
            ### all coref of the gold antecedent except itself, which appeared BEFORE the anaphor
            for child in markables_root:
                markable = child.attrib
                if markable.__contains__('coref_set') and markable['coref_set'] == coref_set:
                    markable_span = markable['span']
                    markable_MinMax_span = self.span_to_MinMax_span(markable_span)
                    # if the coref appeared BEFORE the anaphor
                    if markable_MinMax_span[1] <= anaphor_MinMax_span[0]:
                        surface_str = self.get_tokenized_surface_strings(f_name, markable_MinMax_span,
                                                                         get_tokenized_strings=True,
                                                                         get_surface_string=False)
                        head_infos = self.find_the_head(markable_span, f_name)
                        lemma_list, pos_list, number_list = self.get_words_attributes_list(markable_span, f_name)
                        coref_chains[markable['id']] = {'surface_str': surface_str, 'lemma': lemma_list, 'pos': pos_list,
                                                        'span': markable['span'], 'number': number_list, 'head': head_infos}

            return coref_chains'''

    def normalization(self, tokenized_sentences):
        new_tokenized_sentences = []
        p = inflect.engine()
        self.nlp.tokenizer = self.custom_spacy_tokenizer
        doc = self.nlp(tokenized_sentences)

        for word in doc:
            new_word = word.lemma_.lower()
            new_tokenized_sentences.append(new_word)

        return new_tokenized_sentences

    def find_the_context(self, anaphor_span, x_sents_before, f_name):
        '''To find the idx of the context sentences:
        1. x sentences before the anaphor sentences and
        2. anaphor sentences itself
        @param anaphor_span:
        str: a span from the doc e.g. 'word_5..word_7'
        @param x_sents_before:
        int: x sentences before the anaphor sentences
        @param f_name:
        str: the file name
        @return:
        '''
        ### to retrieve the idxs of the context sentences and the anaphor sentence
        sents_tree = ET.parse(self.corpus_path + "original/sentences/" + f_name)
        whole_context_idxs = {}
        anaphor_MinMax_span = self.span_to_MinMax_span(anaphor_span)
        sents_root = sents_tree.getroot()
        sents_idxs = [self.span_to_MinMax_span(child.attrib['span']) for child in sents_root]  # [[1,19], [20,40], ...]
        cont_sents_MinMax_idxs = None
        ana_sent_MinMax_idx = None

        for sent_idx in sents_idxs:
            if (sent_idx[0] <= anaphor_MinMax_span[0]) and (
                    sent_idx[1] >= anaphor_MinMax_span[1]):  # if the anaphor is in this sentence
                Xth_sent = sents_idxs.index(
                    sent_idx)  # (start wiz 0)in Xth sentence appears the anaphor(index of sents_idxs: sents_idx[0])
                if Xth_sent == 0:
                    cont_sents_MinMax_idxs = [sents_idxs[0]]
                    ana_sent_MinMax_idx = sents_idxs[0]
                elif Xth_sent < x_sents_before:  # if this sentence has no x sentences before, then we take from the beginning to this sentences as the context
                    cont_sents_MinMax_idxs = sents_idxs[:Xth_sent]
                    ana_sent_MinMax_idx = sents_idxs[Xth_sent]
                else:
                    cont_sents_MinMax_idxs = sents_idxs[Xth_sent - x_sents_before:Xth_sent]
                    ana_sent_MinMax_idx = sents_idxs[Xth_sent]

        # the ACTUAL CONTECT IDX SLIDES of the context sentences and the anapor sentence
        first_cont_sent_MinMax_idx = cont_sents_MinMax_idxs[
            0]  # the minmax idx of the first sentence of the context sentences
        last_cont_sent_MinMax_idx = cont_sents_MinMax_idxs[
            -1]  # ......................last...................................

        cont_sents_MinMax_span = [first_cont_sent_MinMax_idx[0], last_cont_sent_MinMax_idx[1]]
        ana_sent_MinMax_span = [ana_sent_MinMax_idx[0], ana_sent_MinMax_idx[1]]

        cont_sents_idx = [first_cont_sent_MinMax_idx[0] - 1, last_cont_sent_MinMax_idx[
            1]]  # the actual idx of the whole context sentences(except anapor sentence)
        ana_sent_idx = [ana_sent_MinMax_idx[0] - 1, ana_sent_MinMax_idx[1]]  # the actual idx of the anaphor sentence

        whole_context_idxs['cont_sents_span'] = cont_sents_MinMax_span
        whole_context_idxs['ana_sent_span'] = ana_sent_MinMax_span
        whole_context_idxs['cont_sents_idx'] = cont_sents_idx
        whole_context_idxs['ana_sent_idx'] = ana_sent_idx

        tokenized_context_sentences = self.get_tokenized_surface_strings(f_name, cont_sents_MinMax_span,
                                                                         get_tokenized_strings=True,
                                                                         get_surface_string=False)
        tokenized_ana_sentences = self.get_tokenized_surface_strings(f_name, ana_sent_MinMax_span,
                                                                     get_tokenized_strings=True,
                                                                     get_surface_string=False)
        tokenized_whole_sentences = tokenized_context_sentences + tokenized_ana_sentences

        preprocessed_whole_sentences = self.normalization(tokenized_whole_sentences)

        context_info = {'whole_context': {'preprocessed_surface_str': preprocessed_whole_sentences,
                                          'span': 'word_{}..word{}'.format(str(cont_sents_MinMax_span[0]),
                                                                           str(ana_sent_MinMax_span[1]))}}
        return whole_context_idxs, context_info

    def get_tokenized_surface_strings(self, f_name, MinMax_span, get_tokenized_strings=True, get_surface_string=True):
        '''To get the (tokenized) surface string(THE SURFACESRINGS IS NOT COOL FOR CONTEXT SENTENCES)
        @param f_name:
        str: the file name
        @param MinMax_span:
        list: could be MinMax_span of a markable or a context e.g [54, 299]
        @param get_tokenized_strings:
        Booleans: if True, return tokenized strings
        @param get_surface_string:
        Booleans: if True, return surface strings
        @return Tokenized or/and not tokenized surface string
        '''

        words_tree = ET.parse(self.corpus_path + "original/words/" + f_name)
        words_root = words_tree.getroot()
        words_string = [child.text for child in
                        words_root]  # the whole document text in a list:['I', 'am', 'good', ...]
        if get_tokenized_strings == True:
            tokenized = words_string[MinMax_span[0] - 1: MinMax_span[1]]
            if get_surface_string == True:
                surface_string = " ".join(tokenized)
                return tokenized, surface_string
            else:
                return tokenized
        elif get_surface_string == True:
            surface_string = " ".join(words_string[MinMax_span[0] - 1: MinMax_span[1]])
            return surface_string
        else:
            print(
                "Error: At least one of the argument \'get_tokenized_strings\' or \'get_surface_string\' of the function \'get_tokenized_surface_strings\' has to be True.")

    def find_the_head(self, phrase_span, f_name):
        '''Find a head of a phrase.
        @param phrase_span: a span of a phrase which we need to find its head e.g. 'word_5..word_7'
        @param f_name: the file name
        @return: the informations of the head --- surface srr, span, pos
        '''
        phrase_MinMax_span = self.span_to_MinMax_span(phrase_span)
        tokenized, surface_string = self.get_tokenized_surface_strings(f_name, phrase_MinMax_span,
                                                                       get_tokenized_strings=True,
                                                                       get_surface_string=True)
        doc = self.nlp(tokenized)
        dep_list, children_list = self.get_dep_list_from_word_file(phrase_span, f_name)
        tok_id = 0
        for token, dep, children in zip(doc, dep_list, children_list):
            token_idx = phrase_MinMax_span[0] + tok_id
            token_span = str('word_' + str(token_idx) + '..word_' + str(token_idx))
            ### find the noun head of the phrase
            if token.dep_ == 'ROOT':
                head_str = token.text
                head_pos = token.pos_
                head_span = token_span
                head_dep = dep
                head_lemma = self.normalization([head_str])[0]
                return {'surface_str': head_str, 'lemma': head_lemma, 'span': head_span, 'pos': head_pos,
                        'dep': head_dep, 'children': children}
            tok_id += 1

    def get_dep_list_from_word_file(self, span, f_name):
        MinMax_span = self.span_to_MinMax_span(span)
        word_id_list = ['word_' + str(i) for i in range(MinMax_span[0], MinMax_span[1] + 1)]
        # find word xml element
        words_tree = ET.parse(self.corpus_path + "modified/words_/" + f_name)
        words_root = words_tree.getroot()
        word_marks_list = []
        for word_id in word_id_list:
            for child in words_root:
                word_markable = child.attrib
                if word_id == word_markable['id']:
                    word_marks_list.append(word_markable)
        dep_list = []
        children_list = []
        for word_markable in word_marks_list:
            dep_list.append(word_markable['dep'])
            children_list.append(word_markable['children'])

        return dep_list, children_list

    def get_words_attributes_list(self, span, f_name):
        MinMax_span = self.span_to_MinMax_span(span)
        word_id_list = ['word_' + str(i) for i in range(MinMax_span[0], MinMax_span[1] + 1)]
        # find word xml element
        words_tree = ET.parse(self.corpus_path + "modified/words_/" + f_name)
        words_root = words_tree.getroot()
        word_marks_list = []
        for word_id in word_id_list:
            for child in words_root:
                word_markable = child.attrib
                if word_id == word_markable['id']:
                    word_marks_list.append(word_markable)
        lemma_list = []
        pos_list = []
        number_list = []
        for word_markable in word_marks_list:
            lemma_list.append(word_markable['lemma'])
            pos_list.append(word_markable['pos'])
            if word_markable.__contains__('number'):
                number_list.append(word_markable['number'])
            else:
                number_list.append(None)

        return lemma_list, pos_list, number_list

    def custom_spacy_tokenizer(self, text: List[str]) -> Doc:
        return Doc(self.nlp.vocab, words=text)

    def enrich_word_corpus(self):
        '''Tag NUMBER(if NP), POS, dependency and lemma in words file, save the new modified corpus'''
        self.nlp.tokenizer = self.custom_spacy_tokenizer
        print('Tagging corpus...')
        # TODO NP = ['NN', 'NNP', 'PRP', ...]
        file_names = listdir(self.corpus_path + "original/words/")
        for f_name in file_names:
            words_tree = ET.parse(self.corpus_path + "original/words/" + f_name)
            words_root = words_tree.getroot()
            sents_tree = ET.parse(self.corpus_path + "original/sentences/" + f_name)
            sents_root = sents_tree.getroot()
            all_tokens_infos = OrderedDict()
            w_id = 1  # first word every file
            # go through all sentences of one file
            for sent in sents_root:
                sent_span = sent.attrib['span']
                MinMax_span = self.span_to_MinMax_span(sent_span)
                tokenized_sent = self.get_tokenized_surface_strings(f_name, MinMax_span, get_tokenized_strings=True,                                                            get_surface_string=False)
                sent_doc = self.nlp(tokenized_sent)
                # get infos of all tokens(whole text)
                for token in sent_doc:
                    token_infos = {}
                    word_id = 'word_' + str(w_id)
                    token_infos['pos'] = token.tag_
                    token_infos['lemma'] = token.lemma_
                    token_infos['surface_str'] = token.text
                    token_infos['dep'] = token.dep_
                    number_list = token.morph.get('Number')
                    token_infos['children'] = [child for child in token.children]

                    if len(number_list) > 0:
                        token_infos['number'] = number_list[0]
                    all_tokens_infos[word_id] = token_infos
                    w_id += 1

            ### TODO tag NUMBER(if NP) and POS in words file
            for word, token_key in zip(words_root.iter('word'), all_tokens_infos):  # token_key: word_id e.g. word_1
                token_surface_str = all_tokens_infos[token_key]['surface_str']
                token_pos = all_tokens_infos[token_key]['pos']
                token_lemma = all_tokens_infos[token_key]['lemma']
                token_dep = all_tokens_infos[token_key]['dep']
                token_children = all_tokens_infos[token_key]['children']
                if str(word.text) == token_surface_str:
                    word.set('pos', token_pos)
                    word.set('lemma', token_lemma)
                    word.set('dep', token_dep)
                    word.set('children', str(token_children))

                    if all_tokens_infos[token_key].__contains__('number'):
                        token_number = all_tokens_infos[token_key]['number']
                        word.set('number', token_number)
                else:
                    print('Wrong index matching of word and token: ')
            save_to_path = '{}{}/words_/{}'.format(self.corpus_path, 'modified', f_name)
            words_tree.write(save_to_path)
        print('Finished')

    def fix_markables_spans_with_comma(self):
        markables_folder_path = self.corpus_path + "original/markables/"
        file_names = listdir(markables_folder_path)

        for f_name in file_names:
            markables_tree = ET.parse(markables_folder_path + f_name)
            markables_root = markables_tree.getroot()
            for markable in markables_root.iter('{www.eml.org/NameSpaces/phrase}markable'):
                markable_span = markable.attrib['span']

                if ',' in str(markable_span):
                    span = re.findall(r'\d+', markable_span)
                    span.sort()
                    span = "word_{}..word_{}".format(span[0], span[-1])
                    markable.set('span', span)

            save_to_path = '{}{}{}'.format(self.corpus_path, 'modified/markables_/', f_name)
            markables_tree.write(save_to_path)

    def fix_markables_spans_switch_start_and_end(self):
        old_markables_folder_path = self.corpus_path + "modified/markables_/"
        file_names = listdir(old_markables_folder_path)

        for f_name in file_names:
            markables_tree = ET.parse(old_markables_folder_path + f_name)
            markables_root = markables_tree.getroot()
            for markable in markables_root.iter('{www.eml.org/NameSpaces/phrase}markable'):
                markable_span = markable.attrib['span']
                MinMax_span = self.span_to_MinMax_span(markable_span)

                if MinMax_span[0] > MinMax_span[1]:
                    MinMax_span = "word_{}..word_{}".format(MinMax_span[1], MinMax_span[0])
                    markable.set('span', MinMax_span)

            save_to_path = '{}{}{}'.format(self.corpus_path, 'modified/markables_/', f_name)
            markables_tree.write(save_to_path)

    def extraction(self):
        '''Extract all important informations in a dataframe and save it into a .csv file
        With column: file_name, anaphor, gold_antecedents, potential_antecedent
        '''
        print("Going through files....")
        for f_name in tqdm(self.file_names):  # each annotation, sents, words document
            annotation_tree = ET.parse(self.corpus_path + "original/annotations/" + f_name)
            annotation_root = annotation_tree.getroot()
            # go through ANNOTATION files(ANAPHORS)
            for child in annotation_root:
                # create a raw in the df for every annaphor
                anaphor_markable = child.attrib
                df_row = {'file_name': None, 'anaphor': {}, 'gold_antecedents': {}, 'potential_antecedent': None,
                          'context': {}}

                anaphor_id = anaphor_markable['id']
                anaphor_span = anaphor_markable['span']
                anaphor_MinMax_span = self.span_to_MinMax_span(anaphor_span)
                anaphor_surface_string = self.get_tokenized_surface_strings(f_name, anaphor_MinMax_span,
                                                                            get_tokenized_strings=True,
                                                                            get_surface_string=False)
                try:
                    ana_coref_set = anaphor_markable['coref_set']
                except KeyError:
                    ana_coref_set = None

                # print(anaphor_markable.keys())
                # anaphor_coref_set = anaphor_markable['coref_set']
                whole_context_idxs, context_info = self.find_the_context(anaphor_span, 2, f_name)

                head_infos = self.find_the_head(anaphor_span, f_name)
                lemma_list, pos_list, number_list = self.get_words_attributes_list(anaphor_span, f_name)
                # ANAPHOR
                df_row['file_name'] = f_name
                df_row['anaphor'][anaphor_id] = {'surface_str': anaphor_surface_string,
                                                 'lemma': lemma_list,
                                                 'pos': pos_list,
                                                 'number': number_list,
                                                 'span': anaphor_span,
                                                 'head': head_infos,
                                                 'coref_set': ana_coref_set}
                # CO-REFERENCES
                gold_id = anaphor_markable['comp_from'].split(";")[0]
                # coref_chain = self.get_coref_chain(gold_id, f_name, anaphor_span)
                # df_row['coref_chain'] = coref_chain

                # CONTEXT
                df_row['context'] = context_info
                df_row['potential_antecedent'] = self.find_potential_antecedents(anaphor_span, f_name,
                                                                                 whole_context_idxs)

                gold_markable = self.get_markable_through_id(gold_id, f_name)
                gold_span = gold_markable['span']
                gold_MinMax_span = self.span_to_MinMax_span(gold_span)
                head_infos = self.find_the_head(gold_span, f_name)
                gold_surface_str = self.get_tokenized_surface_strings(f_name, gold_MinMax_span,
                                                                      get_tokenized_strings=True,
                                                                      get_surface_string=False)
                lemma_list, pos_list, number_list = self.get_words_attributes_list(gold_span, f_name)
                # gold_coref_chain = self.get_coref_chain(gold_id, f_name, gold_span)

                try:
                    gold_coref_set = gold_markable['coref_set']
                except KeyError:
                    gold_coref_set = None

                gold_info = {'span': gold_span, 'surface_str': gold_surface_str, 'lemma': lemma_list,
                             'head': head_infos, 'pos': pos_list, 'number': number_list, 'coref_set': gold_coref_set}
                df_row['gold_antecedents'][gold_id] = gold_info

                self.df_annotations = self.df_annotations.append(df_row, ignore_index=True)

        save_to_path = self.corpus_path + 'preprocessed/annotation_retrieval.csv'
        self.df_annotations.to_csv(save_to_path, sep='\t')
        print('Successfully extracted data!')

    def main(self):
        # self.enrich_word_corpus()
        # self.fix_markables_spans_with_comma()
        # self.fix_markables_spans_switch_start_and_end()
        self.extraction()


if __name__ == '__main__':
    retriever = Retriever('../../corpus/Dataset_comparative_anaphora_resolution/')
    retriever.main()