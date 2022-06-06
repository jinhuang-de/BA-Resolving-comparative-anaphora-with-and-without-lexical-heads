from typing import *
import torch.nn as nn
import allennlp.modules.elmo as E
from loader import *
import os
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from sentence_transformers import SentenceTransformer # phrase-BERT

class Model(nn.Module):
    def __init__(self,
                 batch_size: int,
                 num_layers: int = 1,
                 if_bert=False,
                 distance_feature=False,
                 grammar_role_feature=False,
                 definiteness_feature=False,
                 match_feature=False,
                 synonym_feature=False,
                 hypernym_feature=False,
                 device: torch.DeviceObjType = 'cuda' if torch.cuda.is_available() else 'cpu'):  # input_size: int => ocab_size

        super(Model, self).__init__()
        self.whole_corpus = Corpus().corpus_total
        self.device = device
        self.distance_feature = distance_feature
        self.grammar_feature = grammar_role_feature
        self.definiteness_feature = definiteness_feature
        self.match_feature = match_feature
        self.synonym_feature = synonym_feature
        self.hypernym_feature = hypernym_feature
        self.dropout = nn.Dropout(p=0.2).to(self.device)

        # phrase-BERT EMBEDDINGS
        self.if_bert = if_bert
        if self.if_bert:
            self.bert_model = SentenceTransformer("whaleloops/phrase-bert").to(self.device)

        # ELMo EMBEDDINGS
        elmo_options_file = '../embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
        elmo_weight_file = '../embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
        self.elmo = E.Elmo(elmo_options_file, elmo_weight_file, num_output_representations=1,
                           requires_grad=True).to(self.device)  # , dropout=0.1

        self.elmo_size = 1024
        self.bert_size = 768
        self.embedding_size = self.elmo_size
        self.distance_feature_size = 9
        self.grammar_feature_size = 20
        self.definiteness_feature_size = 6

        self.batch_size = batch_size
        self.num_layers = num_layers
        self.concat = 2
        self.ffnn_input_size = self.elmo_size

        if self.if_bert:
            self.ffnn_input_size += self.bert_size * self.concat

        if self.distance_feature == True:
            self.ffnn_input_size += self.distance_feature_size
        if self.grammar_feature == True:
            self.ffnn_input_size += self.grammar_feature_size
        if self.definiteness_feature == True:
            self.ffnn_input_size += self.definiteness_feature_size
        if self.match_feature == True:
            self.ffnn_input_size += 1
        if self.synonym_feature == True:
            self.ffnn_input_size += 1
        if self.hypernym_feature == True:
            self.ffnn_input_size += 1

        self.ffnn_last_hidden_size = 64
        self.sigmoid = nn.Sigmoid().to(self.device)
        self.max_pooling = nn.MaxPool1d(kernel_size=2).to(self.device)

        # LAYERNORM
        # self.layernorm1 = torch.nn.LayerNorm(round(self.ffnn_input_size / 2))
        # self.layernorm2 = torch.nn.LayerNorm(round(self.ffnn_input_size / 4))
        # self.layernorm3 = torch.nn.LayerNorm(64)

        self.ffnn = nn.Sequential(nn.Linear(self.ffnn_input_size, 512),
                                  self.dropout,
                                  # self.layernorm1,
                                  nn.ReLU(),
                                  nn.Linear(512, 128),
                                  self.dropout,
                                  # self.layernorm2,
                                  nn.ReLU(),
                                  nn.Linear(128, 64),
                                  self.dropout,
                                  # self.layernorm3,
                                  nn.ReLU(),
                                  nn.Linear(64, 1)).to(self.device)


    def get_phraseBERT_embeddings(self, batch_docs):
        '''Return the phrase embeddings of the batch(eatch phrase/mention has one embedding)
        @param samples: list of samples => list of list of tokens
        @return: size(batch_size, mex_sample_len, 1024)
        '''
        batch_anaphor_phrases_str = [' '.join(doc["tokens"]) for doc in batch_docs]
        batch_gold_phrases_str = [' '.join(doc["gold"]["tokens"]) for doc in batch_docs]

        batch_anaphor_span_embeddings = self.bert_model.encode(batch_anaphor_phrases_str)
        batch_gold_span_embeddings = self.bert_model.encode(batch_gold_phrases_str)

        batch_candidates_span_embeddings = []
        for doc in batch_docs:
            candidates_span_embeddings_for_each_ana = [self.bert_model.encode(' '.join(cand['tokens'])) for cand in
                                                       doc["potential_antecedents"]]
            batch_candidates_span_embeddings.append(
                [torch.from_numpy(cand) for cand in candidates_span_embeddings_for_each_ana])

        batch_anaphor_span_embeddings = [torch.from_numpy(ana_r) for ana_r in batch_anaphor_span_embeddings]
        batch_gold_span_embeddings = [torch.from_numpy(g_r) for g_r in batch_gold_span_embeddings]

        # torch.from_numpy(
        return batch_anaphor_span_embeddings, batch_gold_span_embeddings, batch_candidates_span_embeddings


    def get_context_elmo_embeddings(self, batch_docs):
        '''Return the ELMo embeddings of the batch(each token of the mention has one embedding)
        @param batch_docs: list of samples => list of list of tokens
        @return: size(batch_size, mex_sample_len, 1024)
        '''
        # Compute two different representation for each token.
        # Each representation is a linear weighted combination for the
        # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
        batch_context = [doc["context"] for doc in batch_docs]  # 3 sentences together
        character_ids = E.batch_to_ids(batch_context).to(self.device)
        batch_emb = self.elmo(character_ids)['elmo_representations'][0]  # .to(self.device) # tested: noprob
        return batch_emb.to(self.device)

    def get_idxs(self, batch_doc):
        '''Get the idx of the anaphor and its candidates antecedents in its context(to extract thier embeddinfs)
        @param doc: a batch of contexts
        @return: 3 lists---the anaphor slices, the slices of its candidates antecedents and the gold slices of a batch
        '''
        anaphor_slices = []
        potential_slices = []
        gold_slices = []

        for doc in batch_doc:
            mention = doc
            first_index_of_context = mention["first_index_of_context"]
            anaphor_slice = [mention["left"] - first_index_of_context, mention["right"] - first_index_of_context]
            potential_slice = [[p["left"] - first_index_of_context, p["right"] - first_index_of_context] for p in
                               mention["potential_antecedents"]]
            gold_slice = [mention["gold"]["left"] - first_index_of_context,
                          mention["gold"]["right"] - first_index_of_context]

            anaphor_slices.append(anaphor_slice)
            potential_slices.append(potential_slice)
            gold_slices.append(gold_slice)

        return anaphor_slices, potential_slices, gold_slices

    def get_batch_neural_span_of_anaphor(self, batch_lstm_out: torch.Tensor, idx: List):
        '''Using the idx to extract the elmo neural representation of the anaphors from the elmo embeddings of the context
        @param batch_lstm_out: torch.Tensor LSTM output of a batch, size [16, x, 1024]
        @param idx: List[List[int,int]]: a list of slices
        @return: list of Tensors - span representations of one batch
        '''
        neural_spans = []
        for sample_id, _ in enumerate(batch_lstm_out):
            span_idx = idx[sample_id]
            out = batch_lstm_out[sample_id]
            left = span_idx[0]
            right = span_idx[1]
            span_repr = out[left:right, :]
            neural_spans.append(span_repr)

        return neural_spans

    def get_batch_neural_span_of_gold(self, batch_lstm_out: torch.Tensor,
                                      idx: List):
        '''Using the idx to extract the elmo neural representation of the gold mention from the elmo embeddings of the context
        @param batch_lstm_out: torch.Tensor LSTM output of a batch, size [16, seq length, 1024]
        @param idx: List[List[int,int]]: a list of slices
        @return: list of Tensors - span representations of one batch
        '''
        neural_spans = []
        for sample_id, _ in enumerate(batch_lstm_out):
            span_idx = idx[sample_id]
            out = batch_lstm_out[sample_id]
            left = span_idx[0]
            right = span_idx[1]
            span_repr = out[left:right, :]
            neural_spans.append(span_repr)

        return neural_spans

    def get_batch_neural_span_of_potentials(self, batch_lstm_out: torch.Tensor, potentials_idx: List):
        '''Using the idx to extract the elmo neural representation of the candidate mentons from the elmo embeddings of the context
        @param batch_lstm_out: torch.Tensor LSTM output of a batch, size [16, seq length, 1024]
        @param idx: a list of lists of slices
        @return: a list of lists of Tensors - span representations of one batch
        '''
        batch_neural_spans = []
        for sample_id, out in enumerate(batch_lstm_out):
            p_idxs = potentials_idx[sample_id]
            p_reprs = []
            for p_id in p_idxs:
                left = p_id[0]
                right = p_id[1]
                # Flatten words representations in a span together
                p_repr = out[left:right, :]
                p_reprs.append(p_repr)
            batch_neural_spans.append(p_reprs)

        return batch_neural_spans

    def forward(self, docs) -> List[torch.Tensor]:
        '''
        @param docs: a BACTH of documents, each document is a context, which includes one anaphor(and its antecedent and candidates)
        @return: anaphor_candidant pairs concatenated with feature vectors of each document, size [num_pairs, concat_size] => so each pair is a sample
        '''
        elmo_embeddings = self.get_context_elmo_embeddings(docs)  # (batch_size, mex_sample_len, 1026)
        anaphor_slices, potential_slices, gold_slices = self.get_idxs(docs)

        # SPAN EXTRACTION
        # using the idxs to extract the elmo embeddings of the mentions from the elmo embeddings of th context
        batch_anaphor_repr = self.get_batch_neural_span_of_anaphor(elmo_embeddings, anaphor_slices)
        batch_potential_repr = self.get_batch_neural_span_of_potentials(elmo_embeddings, potential_slices)

        # POOLING + SUM
        batch_anaphor_repr = [self.max_pooling(ana_repr.unsqueeze(0)).squeeze(0) for ana_repr in batch_anaphor_repr]
        batch_anaphor_repr = [torch.sum(ana_repr, 0) for ana_repr in batch_anaphor_repr]
        sumed_batch_potential_repr = []
        for sample in batch_potential_repr:
            p_reprs_pro_sample = []
            for p_repr in sample:
                new_p_repr = self.max_pooling(p_repr.unsqueeze(0)).squeeze(0)
                new_p_repr = torch.sum(new_p_repr, 0)
                p_reprs_pro_sample.append(new_p_repr)
            sumed_batch_potential_repr.append(p_reprs_pro_sample)
        batch_potential_repr = sumed_batch_potential_repr

        # concat extracted elmo span representation with phraseBERT representation
        if self.if_bert:
            bert_batch_anaphor_repr, _, bert_batch_potential_repr = self.get_phraseBERT_embeddings(docs)
            batch_anaphor_repr = torch.stack([torch.squeeze(
                torch.cat((torch.unsqueeze(r, 0).to(self.device), torch.unsqueeze(b, 0).to(self.device)), 1).to(
                    self.device), 0) for r, b in zip(batch_anaphor_repr, bert_batch_anaphor_repr)]).to(self.device)

            cat_batch_potential_repr = []
            for ana_p, ana_bert_p in zip(batch_potential_repr,
                                         bert_batch_potential_repr):  # batch_potential_repr: len = batch_size
                p_reprs_pro_sample = []
                for p_repr, bert_p_repr in zip(ana_p, ana_bert_p):
                    new_p_repr = torch.squeeze(torch.cat(
                        (torch.unsqueeze(p_repr, 0).to(self.device), torch.unsqueeze(bert_p_repr, 0).to(self.device)),
                        1).to(self.device), 0).to(self.device)
                    p_reprs_pro_sample.append(new_p_repr)
                cat_batch_potential_repr.append(p_reprs_pro_sample)
            batch_potential_repr = cat_batch_potential_repr

        # PAIRS REPRESENTATIONS
        # NEW size: [512 + 768 + 9 = 1289]
        batch_concat_pairs = []

        # BATCH FEATURES
        batch_distance_features_matrixs = get_batch_distance_features_matrixs(docs)
        batch_grammar_features_matrixs = get_batch_grammar_features_matrixs(docs)
        batch_definiteness_features_matrixs = get_batch_definiteness_features_matrixs(docs)
        batch_match_features_scores = get_batch_match_features_scores(docs)
        batch_synonym_features_scores = get_batch_synonym_features_scores(docs)
        batch_hypernym_features_scores = get_batch_hypernym_features_scores(docs)

        # CONCATENATE ALL ANAPHOR SPAN REPRESENTATION WITH CANDIDATES SPAN REPRESENTATIONS AND CANDIDATES FEATURES
        for ana_id, ana_repr in enumerate(batch_anaphor_repr):
            doc_anaphor_potentials_concat = []
            potential_repr = batch_potential_repr[ana_id]
            # FEATURES PER ANAPHOR
            distance_features_matrixs = batch_distance_features_matrixs[ana_id]
            grammar_features_matrixs = batch_grammar_features_matrixs[ana_id]
            definiteness_features_matrixs = batch_definiteness_features_matrixs[ana_id]

            match_features_scores = batch_match_features_scores[ana_id]
            synonym_features_scores = batch_synonym_features_scores[ana_id]
            hypernym_features_scores = batch_hypernym_features_scores[ana_id]

            for pot_repr, pot_dis_feature_vec, pot_gramm_feature_vec, pot_definitness_feature_vec, match_features_score, synonym_features_score, hypernym_features_score in zip(
                    potential_repr, distance_features_matrixs, grammar_features_matrixs, definiteness_features_matrixs,
                    match_features_scores, synonym_features_scores, hypernym_features_scores):

                # concatenate anaphor and candidates span representation
                ana_cat_pot = torch.cat((torch.squeeze(ana_repr), torch.squeeze(pot_repr)), dim=0).to(self.device)

                # concatenate the embedding concatenation and the feature vectors
                if self.distance_feature == True:
                    pot_dis_feature_vec = torch.squeeze(torch.Tensor(pot_dis_feature_vec)).to(self.device)
                    ana_cat_pot = torch.cat((ana_cat_pot, pot_dis_feature_vec), dim=0).to(self.device)
                if self.grammar_feature == True:
                    pot_gramm_feature_vec = torch.squeeze(torch.Tensor(pot_gramm_feature_vec)).to(self.device)
                    ana_cat_pot = torch.cat((ana_cat_pot, pot_gramm_feature_vec), dim=0).to(self.device)
                if self.definiteness_feature == True:
                    pot_definitness_feature_vec = torch.squeeze(torch.Tensor(pot_definitness_feature_vec)).to(
                        self.device)
                    ana_cat_pot = torch.cat((ana_cat_pot, pot_definitness_feature_vec), dim=0).to(self.device)
                if self.match_feature == True:
                    ana_cat_pot = torch.cat((ana_cat_pot, torch.Tensor([match_features_score]).to(self.device)),
                                            dim=0).to(self.device)
                if self.synonym_feature == True:
                    ana_cat_pot = torch.cat((ana_cat_pot, torch.Tensor([synonym_features_score]).to(self.device)),
                                            dim=0).to(self.device)
                if self.hypernym_feature == True:
                    ana_cat_pot = torch.cat((ana_cat_pot, torch.Tensor([hypernym_features_score]).to(self.device)),
                                            dim=0).to(self.device)

                doc_anaphor_potentials_concat.append(ana_cat_pot)
            batch_concat_pairs.append(doc_anaphor_potentials_concat)

        return batch_concat_pairs

    def predict(self, batch_concat_pairs):
        '''Make prediction: if the candidate is the antecedent of the anaphor or not(binary classification)
        @param batch_concat_pairs: anaphor_candidate pairs, size [num_pairs, concat_size] => each anaphor-candidate pair is a sample!
        @return: the sigmoid score and the binary result
        '''
        # FFNN PREDICTION
        results_sigmoid = []
        results_labels = []
        for pairs_pro_sample in batch_concat_pairs:
            sigmoid_results_pro_sample = []
            labels_reults_pro_sample = []
            for pair in pairs_pro_sample:
                ffnn_out = self.ffnn(pair).to(self.device)
                score = self.sigmoid(ffnn_out)
                sigmoid_results_pro_sample.append(score)  # different len because of different amount of potentials: list of potentials([512])
                labels_reults_pro_sample.append(assign_label(score))
            results_sigmoid.append(sigmoid_results_pro_sample)
            results_labels.append(labels_reults_pro_sample)
        return results_sigmoid, results_labels