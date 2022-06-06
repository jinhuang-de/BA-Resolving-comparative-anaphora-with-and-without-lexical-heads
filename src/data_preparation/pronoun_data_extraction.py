import re
import pandas as pd
import spacy
from spacy.tokens import Doc
from typing import *
import os
from tqdm import tqdm

# TODO: Problem:
#  1. some golds are after anaphor
#  2. head index is wrong


# class Retriever:
class ontonote_prounoun_extraction():
    def __init__(self):
        # LOCAL
        # self.corpus_path = '../../corpus/ontonotes-5.0-conll-2012/conll-2012/v4/data/train/data/english/annotations/'
        # CLUSTER
        self.corpus_path = '../../corpus/ontonotes-5.0-conll-2012/annotations/'
        self.pronoun_tags = ['PRP', 'PRP$']
        self.row = None  # a new row to append on
        self.df = pd.DataFrame(columns=['file_name'])
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.tokenizer = self.custom_spacy_tokenizer

    def get_sent_surface_str(self, sent):
        sent_surface_str = []
        for token_line in sent:
            sent_surface_str.append(token_line[3])
        return sent_surface_str

    def get_candidates_and_gold_mentions(self, context_sents_list, pronoun_idx, pronoun_coref_id):
        candidate_mentions = []
        sent_idx = 0
        cand_mention = {}

        for sent in context_sents_list:
            if sent_idx == len(context_sents_list)-1:
                sent = sent[0:pronoun_idx] # candidates are only mentions before the pronoun in the context

            for token in sent:

                if token[-1] != '-':
                    # find out all the coref ids of this token
                    coref_ids = re.findall(r'\d+', token[-1])
                    for coref_id in coref_ids:  # [(3, (2)|3), (4|(2), 4), (2) ]
                        mention_poses = [token[4]]
                        mention_surface_strings = [token[3]]
                        mention_idxs = [int(token[2])]
                        span_idx_in_context = None

                        # (2)
                        if token[-1] == '(' + str(coref_id) + ')':
                            cand_mention['surface_str'] = mention_surface_strings
                            cand_mention['lemma'] = self.normalization(mention_surface_strings)
                            cand_mention['pos'] = mention_poses
                            cand_mention['coref_id'] = coref_id
                            if sent_idx == 0:
                                span_idx_in_context = [mention_idxs[0], mention_idxs[-1] + 1]
                            elif sent_idx == 1:
                                first_sent_len = len(context_sents_list[0])
                                span_idx_in_context = [first_sent_len + mention_idxs[0], first_sent_len + mention_idxs[-1] + 1]
                            elif sent_idx == 2:
                                first_2_sents_len = len(context_sents_list[0]) + len(context_sents_list[1])
                                span_idx_in_context = [first_2_sents_len + mention_idxs[0], first_2_sents_len + mention_idxs[-1] +1]

                            cand_mention['span_idx_in_context'] = span_idx_in_context

                            cand_mention['head'] = self.find_the_head(mention_surface_strings, span_idx_in_context)

                            candidate_mentions.append(cand_mention)
                            cand_mention = {}

                        elif '(' + str(coref_id) in token[-1]:
                            end_coref_str = str(coref_id) + ')'

                            # check the tokens after this token
                            for token_after in sent[int(token[2]) + 1:]:
                                mention_poses.append(token_after[4])
                                mention_surface_strings.append(token_after[3])
                                mention_idxs.append(int(token_after[2]))
                                if end_coref_str in token_after[-1]:
                                    cand_mention['surface_str'] = mention_surface_strings
                                    cand_mention['lemma'] = self.normalization(mention_surface_strings)
                                    cand_mention['pos'] = mention_poses
                                    cand_mention['coref_id'] = coref_id
                                    if sent_idx == 0:
                                        span_idx_in_context = [mention_idxs[0], mention_idxs[-1] + 1]

                                    elif sent_idx == 1:
                                        first_sent_len = len(context_sents_list[0])
                                        span_idx_in_context = [first_sent_len + mention_idxs[0],
                                                                            first_sent_len + mention_idxs[-1] + 1]
                                    elif sent_idx == 2:
                                        first_2_sents_len = len(context_sents_list[0]) + len(context_sents_list[1])
                                        span_idx_in_context = [first_2_sents_len + mention_idxs[0],
                                                                            first_2_sents_len + mention_idxs[-1] + 1]

                                    cand_mention['span_idx_in_context'] = span_idx_in_context
                                    cand_mention['head'] = self.find_the_head(mention_surface_strings, span_idx_in_context)

                                    candidate_mentions.append(cand_mention)
                                    candidate_mentions.append(cand_mention)
                                    cand_mention = {}
                                    break
            sent_idx += 1

        #Gold
        gold_mentions = []
        for mention in candidate_mentions:
            if mention['coref_id'] == pronoun_coref_id:
                gold_mentions.append(mention)

        unique_cands = []
        for idx, cand in enumerate(candidate_mentions):
            if cand not in unique_cands:
                unique_cands.append(cand)

        unique_golds = []
        for idx, gold in enumerate(gold_mentions):
            if gold not in unique_golds:
                unique_golds.append(gold)

        # print('ana: ', pronoun_idx + )
        # print('context_sents_len: ', len([token for sent in context_sents_list for token in sent]))

        # for cand in candidate_mentions:
        #     print('cand: ', cand['span_idx_in_context'])
        # for gold in gold_mentions:
        #     print('gold: ', gold['span_idx_in_context'])

       # print('\n')

        return unique_cands, unique_golds

    def custom_spacy_tokenizer(self, text: List[str]) -> Doc:
        return Doc(self.nlp.vocab, words=text)

    def normalization(self, tokenized_sentences):
        tokenized_sentences = [t.lower() for t in tokenized_sentences]
        new_tokenized_sentences = []
        self.nlp.tokenizer = self.custom_spacy_tokenizer
        doc = self.nlp(tokenized_sentences)
        for word in doc:
            new_word = word.lemma_
            new_tokenized_sentences.append(new_word)
        return new_tokenized_sentences

    def mention_extraction(self, doc):
        mentions = []
        sent_idx = 0
        mention = {}

        for sent in doc:
            # TODO: APPEND SENTENCE STR DIRECTLY IN LOOP TO EVERY PRONOUN CAUSE IN ONE SENTENCE THERE COULD BE MULTIPLE PRONOUNS
            for token in sent:
                if token[-1] != '-':
                    # find out all the coref ids of this token
                    coref_ids = re.findall(r'\d+', token[-1])
                    for coref_id in coref_ids: # [(3, (2)|3), (4|(2), 4), (2) ]
                        token_idx_in_sent = int(token[2])
                        mention_poses = [token[4]]
                        mention_surface_strings = [token[3]]
                        mention_idxs = [int(token[2])]
                        # (2)
                        if token[-1] == '(' + str(coref_id) + ')':
                            if any(pos in self.pronoun_tags for pos in mention_poses):
                                span_idx_in_context = None
                                left = sent_idx - 2
                                if left < 0:
                                    left = 0
                                context_sentences = doc[left:sent_idx + 1]

                                three_sentence_surface_str = [self.get_sent_surface_str(sent) for sent in
                                                              context_sentences]
                                mention['context'] = {}
                                mention['context']['whole_context'] = {}
                                mention['context']['whole_context']['context_sent_list'] = three_sentence_surface_str
                                context = [word for sent in three_sentence_surface_str for word in sent]
                                mention['context']['whole_context']['preprocessed_surface_str'] = self.normalization(
                                    context)
                                # all coreferences in context
                                mention['potential_antecedents'], mention['gold'] = self.get_candidates_and_gold_mentions(context_sentences,
                                                                                                                          mention_idxs[0],
                                                                                                                          coref_id)
                                if len(context_sentences) == 1:
                                    span_idx_in_context = [mention_idxs[0], mention_idxs[-1] + 1]
                                    mention['span_idx_in_context'] = span_idx_in_context
                                elif len(context_sentences) == 2:
                                    first_sent_len = len(three_sentence_surface_str[0])
                                    span_idx_in_context = [first_sent_len + mention_idxs[0], first_sent_len + mention_idxs[-1] + 1]
                                    mention['span_idx_in_context'] = span_idx_in_context

                                elif len(context_sentences) == 3:
                                    first_2_sents_len = len(three_sentence_surface_str[0]) + len(three_sentence_surface_str[1])
                                    span_idx_in_context = [first_2_sents_len + mention_idxs[0],
                                                                      first_2_sents_len + mention_idxs[-1] + 1]
                                    mention['span_idx_in_context'] = span_idx_in_context
                                mention['head'] = self.find_the_head(mention_surface_strings, span_idx_in_context)

                            mention['surface_str'] = mention_surface_strings
                            mention['lemma'] = self.normalization(mention_surface_strings)
                            mention['pos'] = mention_poses
                            mention['coref_id'] = coref_id

                            mentions.append(mention)
                            mention = {}

                        elif '(' + str(coref_id) in token[-1]:
                            end_coref_str = str(coref_id) + ')'
                            # check the tokens after this token
                            for token_after in sent[int(token[2])+1:]:
                                mention_poses.append(token_after[4])
                                mention_surface_strings.append(token_after[3])
                                mention_idxs.append(int(token_after[2]))
                                if end_coref_str in token_after[-1]:
                                    if any(pos in self.pronoun_tags for pos in mention_poses):
                                        span_idx_in_context = None
                                        left = sent_idx - 2
                                        if left < 0:
                                            left = 0
                                        context_sentences = doc[left:sent_idx + 1]
                                        three_sentence_surface_str = [self.get_sent_surface_str(sent) for sent in context_sentences]
                                        mention['context'] = {}
                                        mention['context']['whole_context'] = {}
                                        mention['context']['whole_context']['context_sent_list'] = three_sentence_surface_str
                                        context = [word for sent in three_sentence_surface_str for word in sent]
                                        mention['context']['whole_context']['preprocessed_surface_str'] = self.normalization(context)
                                        # all coreferences in context
                                        mention['potential_antecedents'], mention['gold'] = self.get_candidates_and_gold_mentions(context_sentences, mention_idxs[0], coref_id)

                                        if len(context_sentences) == 1:
                                            span_idx_in_context = [mention_idxs[0], mention_idxs[-1] + 1]
                                            mention['span_idx_in_context'] = span_idx_in_context
                                        elif len(context_sentences) == 2:
                                            first_sent_len = len(three_sentence_surface_str[0])
                                            span_idx_in_context = [first_sent_len + mention_idxs[0],
                                                                              first_sent_len + mention_idxs[-1] + 1]
                                            mention['span_idx_in_context'] = span_idx_in_context
                                        elif len(context_sentences) == 3:
                                            first_2_sents_len = len(three_sentence_surface_str[0]) + len(
                                                three_sentence_surface_str[1])
                                            span_idx_in_context = [first_2_sents_len + mention_idxs[0],
                                                                              first_2_sents_len + mention_idxs[-1] + 1]
                                            mention['span_idx_in_context'] = span_idx_in_context
                                        mention['head'] = self.find_the_head(mention_surface_strings, span_idx_in_context)

                                    mention['surface_str'] = mention_surface_strings
                                    mention['lemma'] = self.normalization(mention_surface_strings)
                                    mention['pos'] = mention_poses
                                    mention['coref_id'] = coref_id


                                    mentions.append(mention)
                                    mention = {}
                                    break
            sent_idx += 1
        return mentions

    def find_the_head(self, surface_strs: List, span_idx_in_context):
        '''Find a head of a phrase.
        @param phrase_span: a span of a phrase which we need to find its head e.g. 'word_5..word_7'
        @param f_name: the file name
        @return: the informations of the head --- surface srr, span, pos
        '''
        head_infos = {}
        doc = self.nlp(surface_strs)

        head_id = 0
        for token in doc:
            if token.dep_ == 'ROOT':
                head_infos['pos'] = token.tag_
                head_infos['lemma'] = token.lemma_
                head_infos['surface_str'] = token.text
                head_infos['dep'] = token.dep_
                # number_list = token.morph.get('Number')
                head_infos['children'] = [str(child) for child in token.children]
                head_infos['span_idx_in_context'] = [span_idx_in_context[0] + head_id, span_idx_in_context[0] + head_id + 1]

                head_id += 1

        return head_infos

    # TODO: LOOP THROUGH ORDNERS
    def read_conll_file(self, one_conll_file_path):
        with open(one_conll_file_path, 'r') as f:
            data = f.readlines()[1:-2] # list of token line

        data = [re.split(' +', token_line[:-1]) for token_line in data]

        # split rhe whole corpus in to sentences
        all_sentences = []
        sentence = []
        for token in data:
            if '#end' in token or '#begin' in token:
                pass
            elif len(token) != 1: # if its not newline(
                sentence.append(token)
            else:
                all_sentences.append(sentence)
                sentence = [] # next sentence

        return all_sentences

    def pronoun_extraction(self, all_mentions_in_doc):
        all_pronouns = []
        for mention in all_mentions_in_doc:
            if any(pos in self.pronoun_tags for pos in mention['pos']) and len(mention['gold'])>0:
                all_pronouns.append(mention)

        return all_pronouns

    def list_all_file_pathes(self):
        list_of_files = {}
        for (dirpath, dirnames, filenames) in os.walk(self.corpus_path):
            for filename in filenames:
                if filename.endswith('.v4_gold_conll'):
                    list_of_files[filename] = os.sep.join([dirpath, filename])
        return list_of_files


    def main(self):
        all_file_pathes = self.list_all_file_pathes()
        for file_name, file_path in tqdm(all_file_pathes.items()):
            doc = self.read_conll_file(file_path)
            all_mentions_in_doc = self.mention_extraction(doc)
            all_pronouns = self.pronoun_extraction(all_mentions_in_doc)
            for pronoun in all_pronouns:
                df_row = {'file_name': file_name,
                          'anaphor': {},
                          'gold_antecedents': pronoun['gold'],
                          'potential_antecedent': pronoun['potential_antecedents'],
                          'context': pronoun['context']}

                df_row['anaphor']['surface_str'] = pronoun['surface_str']
                df_row['anaphor']['lemma'] = pronoun['lemma']
                df_row['anaphor']['pos'] = pronoun['pos']
                df_row['anaphor']['coref_id'] = pronoun['coref_id']
                df_row['anaphor']['span_idx_in_context'] = pronoun['span_idx_in_context']
                df_row['anaphor']['head'] = pronoun['head']

                self.df = self.df.append(df_row, ignore_index=True)


        save_to_path = '../../corpus/ontonotes-5.0-conll-2012/preprocessed/ontonotes_retrieval.csv'
        self.df.to_csv(save_to_path, sep='\t')
        print('Successfully extracted data!')

if __name__ == '__main__':
    ontonote = ontonote_prounoun_extraction()
    ontonote.main()