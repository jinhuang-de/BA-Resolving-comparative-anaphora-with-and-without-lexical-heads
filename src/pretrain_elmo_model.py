from typing import *
import torch.nn as nn
import allennlp.modules.elmo as E
from loader_ontonotes_pronoun import *
from torch.nn.utils.rnn import pad_sequence
import gensim
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class Model(nn.Module):
    def __init__(self,
                 batch_size: int,
                 num_layers: int = 1,
                 distance_feature=False,
                 grammar_role_feature=False,
                 definiteness_feature=False,
                 match_feature=False,
                 synonym_feature=False,
                 hypernym_feature=False,
                 pretraining=True,
                 device: torch.DeviceObjType = 'cuda' if torch.cuda.is_available() else 'cpu'):  # input_size: int => ocab_size

        super(Model, self).__init__()
        self.whole_corpus = Corpus().corpus_pronoun
        self.pretraining = pretraining
        self.device = device
        self.distance_feature = distance_feature
        self.grammar_feature = grammar_role_feature
        self.definiteness_feature = definiteness_feature
        self.match_feature = match_feature
        self.synonym_feature = synonym_feature
        self.hypernym_feature = hypernym_feature

        self.dropout = nn.Dropout(p=0.2).to(self.device)

        # SIZES
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.concat = 2
        self.elmo_size = 1024 # 256
        self.embedding_size = self.elmo_size

        self.current_batch_size = batch_size

        # FEATURES
        self.distance_feature_size = 9
        self.grammar_feature_size = 20
        self.definiteness_feature_size = 6
        self.ffnn_input_size = self.elmo_size

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

        self.max_pooling = nn.MaxPool1d(kernel_size=2).to(self.device)

        # self.max_pooling = nn.MaxPool1d(kernel_size=2).to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)

        # EMBEDDINGS
        elmo_options_file = '../embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
        elmo_weight_file = '../embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'

        self.elmo = E.Elmo(elmo_options_file, elmo_weight_file, num_output_representations=1, requires_grad=True).to(self.device)  # , dropout=0.1

        # FEED FORWARD NEURAL NETWORK
        # FOR EACH ANAPHOR-CANDIDATE_X PAIR
        # self.layernorm1 = torch.nn.LayerNorm(round(self.ffnn_input_size / 2))
        # self.layernorm2 = torch.nn.LayerNorm(round(self.ffnn_input_size / 4))
        # self.layernorm3 = torch.nn.LayerNorm(64)

        # model 1
        self.ffnn = nn.Sequential(nn.Linear(self.ffnn_input_size, round(self.ffnn_input_size / 2)),
                                  self.dropout,
                                  # self.layernorm1,
                                  nn.ReLU(),
                                  nn.Linear(round(self.ffnn_input_size / 2), round(self.ffnn_input_size / 4)),
                                  self.dropout,
                                  # self.layernorm2,
                                  nn.ReLU(),
                                  nn.Linear(round(self.ffnn_input_size / 4), 64),
                                  self.dropout,
                                  # self.layernorm3,
                                  nn.ReLU(),
                                  nn.Linear(64, 1)).to(self.device)


    def get_context_elmo_embeddings(self, batch_docs):
        '''
        @param samples: list of samples => list of list of tokens
        @return: size(batch_size, mex_sample_len, 1024)
        '''
        # Compute two different representation for each token.
        # Each representation is a linear weighted combination for the
        # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
        batch_context = [doc["context"] for doc in batch_docs]  # 3 sentences together
        character_ids = E.batch_to_ids(batch_context).to(self.device)
        batch_emb = self.elmo(character_ids)['elmo_representations'][0]  # .to(self.device) # tested: noprob
        return batch_emb.to(self.device)  # size: [16, 64, 256] (batch_size, timesteps, embedding_dim)

    # TODO: ANPASSEN FÜR NEUE DATEN
    def get_idxs(self, batch_doc):
        '''Get the idx of the anaphor and its candidates antecedents in its context.
        @param doc: a batch of anaphor objects
        @return: 2 lists --- the anaphor slices and slices of its candidates antecedents of a batch
        '''
        anaphor_slices = []
        potential_slices = []

        for doc in batch_doc:
            mention = doc
            if self.pretraining == True:
                anaphor_slice = [mention["left"], mention["right"]]
                potential_slice = [[p["left"], p["right"]] for p in mention["potential_antecedents"]]
            else:
                first_index_of_context = mention["first_index_of_context"]
                anaphor_slice = [mention["left"] - first_index_of_context, mention["right"] - first_index_of_context]
                potential_slice = [[p["left"] - first_index_of_context, p["right"] - first_index_of_context] for p in mention["potential_antecedents"]]

            anaphor_slices.append(anaphor_slice)
            potential_slices.append(potential_slice)

        return anaphor_slices, potential_slices

    def get_batch_neural_span(self, batch_lstm_out: torch.Tensor, idx: List):  # lstm output 16 x seq length x 512
        '''
        @param batch_lstm_out: torch.Tensor LSTM output of a batch
        @param idx: List[List[int,int]]: a list of slices
        @return: list of Tensors - span representations of one batch
        '''
        neural_spans = []
        # batch_lstm_out: size [# batch, max_seq_len, hidden_size*directions] = [16, x, 512]
        for sample_id, _ in enumerate(batch_lstm_out):
            span_idx = idx[sample_id]  # [1, 5]
            out = batch_lstm_out[sample_id]  # : size[64, 512] => tested: no prob
            left = span_idx[0]
            right = span_idx[1]
            span_repr = out[left:right, :]
            neural_spans.append(span_repr)

        return neural_spans

    def get_batch_neural_span_of_potentials(self, batch_lstm_out: torch.Tensor, potentials_idx: List):  # lstm output 16 x seq length x 512
        '''
        @param batch_lstm_out: torch.Tensor LSTM output of a batch
        @param idx: List[List[List[int,int]]]: a list of list of slices | batch[slice1[], ...]
        @return: list of Tensors - span representations
        '''
        batch_neural_spans = []
        # batch_lstm_out : size [# batch, max_seq_len, hidden_size*directions] = [16, x, 512]
        for sample_id, out in enumerate(batch_lstm_out):
            p_idxs = potentials_idx[sample_id]  # [1, 5]
            p_reprs = []
            for p_id in p_idxs:
                left = p_id[0]
                right = p_id[1]
                # Flatten words representations in a span together
                p_repr = out[left:right, :]
                p_reprs.append(p_repr)
            batch_neural_spans.append(p_reprs)

        return batch_neural_spans  # list batch [list[tensors....]]

    def forward(self, docs) -> List[torch.Tensor]:
        '''
        @param docs: a BACTH of anaphor object
        @param current_batch_size: the current batch size(Every batch size is diffrent because every anaphor has different amount of candidates.)
        @return:
        '''
        elmo_embeddings = self.get_context_elmo_embeddings(docs)  # (batch_size, mex_sample_len, 256)  e.g. [16, 64, 256]
        last_hidden = elmo_embeddings

        anaphor_slices, potential_slices = self.get_idxs(docs)

        # ------------------------FOR DEBUG-------------------------
        # for i in range(len(docs)):
        #     print('anaphor: , ', docs[i]['tokens'])
        #     print('anaphor left and right: ', docs[i]['left'], docs[i]['right'])
        #     # print('first_index_of_context: ', docs[i]['first_index_of_context'])
        #     print('anaphor_slices[i]: ', anaphor_slices[i])
        #     print('context: ', len(docs[i]['context']), '\n', docs[i]['context'])
        #     print('embedding size: ', last_hidden[i].size())
        #     print('\n')
        # --------------------------------------------------------

        # Extract their span representations from the hidden matrix
        batch_anaphor_repr = self.get_batch_neural_span(last_hidden, anaphor_slices)
        batch_potential_repr = self.get_batch_neural_span_of_potentials(last_hidden, potential_slices)

        # maxpooling
        batch_anaphor_repr = [self.max_pooling(ana_repr.unsqueeze(1)) for ana_repr in batch_anaphor_repr]

        # sum the words representations of each span so that every span representation is 1-d
        batch_anaphor_repr = [torch.sum(ana_repr, 0) for ana_repr in batch_anaphor_repr]  # len = batch_size, ana_repr:[[1, 256]]

        sumed_pooled_batch_potential_repr = []
        for sample in batch_potential_repr:# batch_potential_repr: len = batch_size
            p_reprs_pro_sample = []
            for p_repr in sample:
                new_p_repr = self.max_pooling(p_repr.unsqueeze(1)) # before sum [x, 1, 256]
                # Klowersa: sum the words representations of each span so that every span representation is 1-d
                new_p_repr = torch.sum(new_p_repr, 0)
                p_reprs_pro_sample.append(new_p_repr)
            sumed_pooled_batch_potential_repr.append(p_reprs_pro_sample)
        batch_potential_repr = sumed_pooled_batch_potential_repr

        # PAIRS REPRESENTATIONS
        # NEW size: samples[potentials[pot_tenser[512 + 256 + 9 = 777]]]
        batch_concat_pairs = []

        # BATCH FEATURES
        batch_distance_features_matrixs = get_batch_distance_features_matrixs(docs)
        batch_grammar_features_matrixs = get_batch_grammar_features_matrixs(docs)
        batch_definiteness_features_matrixs = get_batch_definiteness_features_matrixs(docs)
        batch_match_features_scores = get_batch_match_features_scores(docs)
        batch_synonym_features_scores = get_batch_synonym_features_scores(docs)
        batch_hypernym_features_scores = get_batch_hypernym_features_scores(docs)

        # CONCATENATE ALL ANAPHOR SPAN REPRESENTATION WITH CANDIDATES SPAN REPRESENTATIONS AND CANDIDATES FEATURES
        for ana_id, ana_repr in enumerate(batch_anaphor_repr):  # ana_repr: size [1, 256]
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

                # concatenate the span concatenation and the distance feature vector
                if self.distance_feature == True:
                    pot_dis_feature_vec = torch.squeeze(torch.Tensor(pot_dis_feature_vec)).to(self.device)
                    ana_cat_pot = torch.cat((ana_cat_pot, pot_dis_feature_vec), dim=0).to(self.device)

                # concatenate the span concatenation and the grammar feature vector
                if self.grammar_feature == True:
                    pot_gramm_feature_vec = torch.squeeze(torch.Tensor(pot_gramm_feature_vec)).to(self.device)
                    ana_cat_pot = torch.cat((ana_cat_pot, pot_gramm_feature_vec), dim=0).to(self.device)

                if self.definiteness_feature == True:
                    pot_definitness_feature_vec = torch.squeeze(torch.Tensor(pot_definitness_feature_vec)).to(self.device)
                    ana_cat_pot = torch.cat((ana_cat_pot, pot_definitness_feature_vec), dim=0).to(self.device)

                if self.match_feature == True:
                    ana_cat_pot = torch.cat((ana_cat_pot, torch.Tensor([match_features_score]).to(self.device)), dim=0).to(self.device)

                if self.synonym_feature == True:
                    ana_cat_pot = torch.cat((ana_cat_pot, torch.Tensor([synonym_features_score]).to(self.device)), dim=0).to(self.device)

                if self.hypernym_feature == True:
                    ana_cat_pot = torch.cat((ana_cat_pot, torch.Tensor([hypernym_features_score]).to(self.device)), dim=0).to(self.device)

                doc_anaphor_potentials_concat.append(ana_cat_pot)
            batch_concat_pairs.append(doc_anaphor_potentials_concat)

        return batch_concat_pairs, potential_slices  # len = batch size :  list of lists of potentials scores(tensors with one float e.g. tensor([0.27]))

    def predict(self, batch_concat_pairs, potential_slices):
        # FFNN PREDICTION
        results_sigmoid = []
        results_labels = []
        for pairs_pro_sample, p_slices_pro_sample in zip(batch_concat_pairs, potential_slices):
            sigmoid_results_pro_sample = []
            labels_reults_pro_sample = []
            for pair in pairs_pro_sample:
                ffnn_out = self.ffnn(pair).to(self.device)  # mat1 and mat2 shapes cannot be multiplied (1x550 and 1150x575)
                score = self.sigmoid(ffnn_out)
                sigmoid_results_pro_sample.append(score)  # different len because of different amount of potentials: list of potentials([512])
                labels_reults_pro_sample.append(assign_label(score))
            results_sigmoid.append(sigmoid_results_pro_sample)
            results_labels.append(labels_reults_pro_sample)
        return results_sigmoid, results_labels