from typing import *
import torch.nn as nn
import allennlp.modules.elmo as E
from loader import *
from torch.nn.utils.rnn import pad_sequence
import gensim
from sentence_transformers import SentenceTransformer

class bilstm_model(nn.Module):
    def __init__(self,
                 batch_size: int,
                 num_layers: int=1,
                 bidirectional: bool=True,
                 glove=True,
                 if_elmo=True,
                 if_bert=False,
                 distance_feature=False,
                 grammar_role_feature=False,
                 definiteness_feature=False,
                 match_feature=False,
                 synonym_feature=False,
                 hypernym_feature=False,
                 device: torch.DeviceObjType = 'cuda' if torch.cuda.is_available() else 'cpu'):# input_size: int => ocab_size

        super(bilstm_model, self).__init__()
        self.whole_corpus = Corpus().corpus_total
        self.device = device
        self.glove=glove
        self.distance_feature = distance_feature
        self.grammar_feature = grammar_role_feature
        self.definiteness_feature = definiteness_feature
        self.match_feature = match_feature
        self.synonym_feature = synonym_feature
        self.hypernym_feature = hypernym_feature
        self.if_elmo = if_elmo
        self.if_bert = if_bert

        # EMBEDDINGS
        if self.if_bert:
            self.bert_model = SentenceTransformer("whaleloops/phrase-bert").to(self.device)

        elmo_options_file = '../embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
        elmo_weight_file = '../embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
        self.elmo = E.Elmo(elmo_options_file, elmo_weight_file, num_output_representations=1)

        self.glove_word_to_idx = None
        self.glove_weight = None
        self.glove_embeddings = None
        if self.glove:
            self.glove_weight = self.get_glove_weight()
            self.glove_embeddings = nn.Embedding.from_pretrained(self.glove_weight).to(self.device)
            self.glove_embeddings.weight.requires_grad = False

        # SIZES
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.elmo_size = 1024
        self.glove_size = 300
        self.bert_size = 768

        self.embedding_size = 0
        if self.glove:
            self.embedding_size = self.embedding_size + self.glove_size
        if self.if_elmo:
            self.embedding_size = self.embedding_size + self.elmo_size

        self.bilstm_input_size = self.embedding_size
        self.bilstm_hidden_size = 256
        self.bidirectional = bidirectional
        self.directions = 2 if self.bidirectional else 1
        self.concat = 2
        self.bidirection = 2
        self.current_batch_size = batch_size

        # FEATURES
        self.distance_feature_size = 9
        self.grammar_feature_size = 20
        self.definiteness_feature_size = 6
        self.ffnn_input_size = self.bilstm_hidden_size * self.bidirection * self.concat# * 0,5 because of max pooling

        if self.if_bert:
            self.ffnn_input_size += self.bert_size * 2

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

        # model
        self.max_pooling = nn.MaxPool1d(kernel_size=2).to(self.device)
        self.dropout = nn.Dropout(p=0.2).to(self.device)
        self.lstm = nn.LSTM(input_size=self.bilstm_input_size,
                            hidden_size=self.bilstm_hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=self.bidirectional)
        self.sigmoid = nn.Sigmoid()

        # FEED FORWARD NEURAL NETWORK(FOR EACH ANAPHOR-CANDIDATE PAIR)
        self.ffnn = nn.Sequential(nn.Linear(self.ffnn_input_size, round(self.ffnn_input_size / 2)),
                                  self.dropout,
                                  # self.layernorm1,
                                  nn.ReLU(),
                                  nn.Linear(round(self.ffnn_input_size / 2), round(self.ffnn_input_size / 4)),
                                  self.dropout,
                                  nn.ReLU(),
                                  nn.Linear(round(self.ffnn_input_size / 4), 64),
                                  self.dropout,
                                  nn.ReLU(),
                                  nn.Linear(64, 1)).to(self.device)


    def _init_hidden(self, current_batch_size) -> Tuple[torch.Tensor]:
        '''Initialize the hidden layers in forward function
        @param current_batch_size: the current batch size(Every batch size is diffrent because every anaphor has different amount of candidates.)
        '''
        h = c = torch.zeros(self.num_layers * self.directions, current_batch_size, self.bilstm_hidden_size)
        self.current_batch_size = current_batch_size
        return h.to(self.device), c.to(self.device)

    def get_phraseBERT_embeddings(self, batch_docs):
        '''
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
            batch_candidates_span_embeddings.append([torch.from_numpy(cand) for cand in candidates_span_embeddings_for_each_ana])

        batch_anaphor_span_embeddings = [torch.from_numpy(ana_r) for ana_r in batch_anaphor_span_embeddings]
        batch_gold_span_embeddings = [torch.from_numpy(g_r) for g_r in batch_gold_span_embeddings]

        return batch_anaphor_span_embeddings, batch_gold_span_embeddings, batch_candidates_span_embeddings

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
        batch_emb = self.elmo(character_ids)['elmo_representations'][0] # tested: noprob
        return batch_emb.to(self.device) # size: [16, 64, 256] (batch_size, timesteps, embedding_dim)

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
            first_index_of_context = mention["first_index_of_context"]
            anaphor_slice = [mention["left"] - first_index_of_context, mention["right"] - first_index_of_context]
            potential_slice = [[p["left"] - first_index_of_context, p["right"] - first_index_of_context] for p in mention["potential_antecedents"]]
            anaphor_slices.append(anaphor_slice)
            potential_slices.append(potential_slice)

        return anaphor_slices, potential_slices

    def get_batch_neural_span_of_anaphor(self, batch_lstm_out: torch.Tensor, idx: List):  # lstm output 16 x seq length x 512
        '''
        @param batch_lstm_out: torch.Tensor LSTM output of a batch
        @param idx: List[List[int,int]]: a list of slices
        @return: list of Tensors - span representations of one batch
        '''
        neural_spans = []
        # batch_lstm_out: size [# batch, max_seq_len, hidden_size*directions] = [16, x, 512]
        for sample_id, _ in enumerate(batch_lstm_out):
            span_idx = idx[sample_id] # [1, 5]
            out = batch_lstm_out[sample_id] # : size[64, 512] => tested: no prob
            left = span_idx[0]
            right = span_idx[1]
            span_repr = out[left:right, :]
            neural_spans.append(span_repr)

        return neural_spans

    def get_batch_neural_span_of_gold(self, batch_lstm_out: torch.Tensor, idx: List):  # lstm output 16 x seq length x 512
        '''
        @param batch_lstm_out: torch.Tensor LSTM output of a batch
        @param idx: List[List[int,int]]: a list of slices
        @return: list of Tensors - span representations of one batch
        '''
        neural_spans = []
        # batch_lstm_out: size [# batch, max_seq_len, hidden_size*directions] = [16, x, 512]
        for sample_id, _ in enumerate(batch_lstm_out):
            span_idx = idx[sample_id] # [1, 5]
            out = batch_lstm_out[sample_id] # : size[64, 512] => tested: no prob
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
            p_reprs= []
            for p_id in p_idxs:
                left = p_id[0]
                right = p_id[1]
                # Flatten words representations in a span together
                p_repr = out[left:right, :]
                p_reprs.append(p_repr)
            batch_neural_spans.append(p_reprs)

        return batch_neural_spans # list batch [list[tensors....]]

    def get_glove_weight(self):
        '''
        @return: the pretrained weight of glove from gensim
        '''
        all_tokens = [word for ana in self.whole_corpus for word in ana.context]
        vocab = set(all_tokens)

        word_to_idx = {word: i + 2 for i, word in enumerate(vocab)}
        word_to_idx['<unk>'] = 0
        word_to_idx['<pad>'] = 1
        idx_to_word = {i + 2: word for i, word in enumerate(vocab)}
        idx_to_word[0] = '<unk>'
        idx_to_word[1] = '<pad>'

        wvmodel = gensim.models.KeyedVectors.load_word2vec_format(
            '../embeddings/glove/glove.840B.300d.txt',
            binary=False, encoding='utf-8')

        vocab_size = len(word_to_idx)
        embed_size = 300
        weight = torch.zeros(vocab_size, embed_size)
        for i in range(len(wvmodel.index_to_key)):
            try:
                index = word_to_idx[wvmodel.index_to_key[i]]
            except KeyError:
                index = word_to_idx[wvmodel.index_to_key[0]]
            weight[index, :] = torch.from_numpy(wvmodel.get_vector(idx_to_word[index]))

        self.glove_word_to_idx = word_to_idx

        return weight

    def get_batch_glove_embeddings(self, docs):
        '''
        @param docs: the anaphot object
        @return: the glove embedding vectors of a batch of samples with size[batch, max_seq, glove_emb_dim]
        '''
        batch_context = [doc['context'] for doc in docs]
        batch_glove_idxs = []
        for context in batch_context:
            context_tensor = []
            for i in context:
                try:
                    context_tensor.append(self.glove_word_to_idx[i])
                except KeyError:
                    context_tensor.append(self.glove_word_to_idx['<unk>'])
            context_glove_idxs = torch.LongTensor(context_tensor)
            batch_glove_idxs.append(context_glove_idxs)

        # PAD the idxs
        batch_glove_idxs_tensor = pad_sequence(batch_glove_idxs, batch_first=True, padding_value=1) # because the index of <pad> is 1

        batch_glove_embeddings = []
        for samp_idx in batch_glove_idxs_tensor:
            sample_idxs = []
            # embed every word
            for word_idx in samp_idx:
                word_embedding = self.glove_embeddings(word_idx.to(self.device)).view(-1).to(self.device) # [300]
                sample_idxs.append(word_embedding)
            batch_glove_embeddings.append(torch.stack(sample_idxs))

        batch_glove_embeddings = torch.stack(batch_glove_embeddings) # [16, 97, 300] [batch, max_seq, glove_emb_dim]

        return batch_glove_embeddings.to(self.device)

    def forward(self, docs, current_batch_size) -> List[torch.Tensor]:
        '''
        @param docs: a BACTH of anaphor object
        @param current_batch_size: the current batch size is the number of the anaphor-candidate pair(Every batch size is diffrent because every anaphor has different amount of candidates.)
        @return:anaphor_candidant pairs concatenated with feature vectors of each document, size [num_pairs, concat_size] => so each pair is a sample
        '''
        elmo_embeddings = self.get_context_elmo_embeddings(docs)  # (batch_size, mex_sample_len, 256)  e.g. [16, 64, 256]

        # Concatenation of ELMo embeddings and Glove embeddings(optionnal)
        if self.if_elmo:
            embeddings = elmo_embeddings
            if self.glove:
                embeddings = my_3d_concat(elmo_embeddings, self.get_batch_glove_embeddings(docs))
        else:
            embeddings = self.get_batch_glove_embeddings(docs)


        h_0, c_0 = self._init_hidden(current_batch_size) #h_0: (num_layers * num_directions, batch, hidden_size)
        h_0.to(self.device)
        c_0.to(self.device)

        out, (h, _) = self.lstm(embeddings, (h_0, c_0))  # Expected hidden[0] size (4, 18, 256), got [2, 16, 256]
        num_directions = 2 if self.lstm.bidirectional else 1
        layer_norm = nn.LayerNorm(out.size()).to(self.device)
        out = layer_norm(out)

        # LSTM output size[ batch * max_seq_len * hidden_size*directions]
        last_hidden = out.view(self.current_batch_size, -1,
                               self.lstm.num_layers * num_directions * self.lstm.hidden_size).to(self.device)

        # Find the Slices of spans mentions in thier context
        anaphor_slices, potential_slices = self.get_idxs(docs)

        # SPAN ESTRACTION
        batch_anaphor_repr = self.get_batch_neural_span_of_anaphor(last_hidden, anaphor_slices)
        batch_potential_repr = self.get_batch_neural_span_of_potentials(last_hidden, potential_slices)

        # no pooling, only sum
        # batch_anaphor_repr = [self.max_pooling(ana_repr.unsqueeze(0)).squeeze(0) for ana_repr in batch_anaphor_repr]
        batch_anaphor_repr = [torch.sum(ana_repr, 0) for ana_repr in batch_anaphor_repr]

        sumed_batch_potential_repr = []
        for sample in batch_potential_repr:
            p_reprs_pro_sample = []
            for p_repr in sample:
                # new_p_repr = self.max_pooling(p_repr.unsqueeze(0)).squeeze(0)  # before sum [x, 1, 256]
                new_p_repr = torch.sum(p_repr, 0)
                p_reprs_pro_sample.append(new_p_repr)
            sumed_batch_potential_repr.append(p_reprs_pro_sample)
        batch_potential_repr = sumed_batch_potential_repr

        # concat extracted span representation with spanBERT representation
        if self.if_bert:
            bert_batch_anaphor_repr, _, bert_batch_potential_repr = self.get_phraseBERT_embeddings(docs)
            batch_anaphor_repr = torch.stack([torch.squeeze(torch.cat((torch.unsqueeze(r, 0).to(self.device), torch.unsqueeze(b, 0).to(self.device)), 1).to(self.device), 0) for r, b in zip(batch_anaphor_repr, bert_batch_anaphor_repr)]).to(self.device)
            cat_batch_potential_repr = []
            for ana_p, ana_bert_p in zip(batch_potential_repr, bert_batch_potential_repr):
                p_reprs_pro_sample = []
                for p_repr, bert_p_repr in zip(ana_p, ana_bert_p):
                    new_p_repr = torch.squeeze(torch.cat((torch.unsqueeze(p_repr, 0).to(self.device), torch.unsqueeze(bert_p_repr, 0).to(self.device)), 1).to(self.device), 0).to(self.device)
                    p_reprs_pro_sample.append(new_p_repr)
                cat_batch_potential_repr.append(p_reprs_pro_sample)
            batch_potential_repr = cat_batch_potential_repr

        # PAIRS REPRESENTATIONS
        batch_concat_pairs = []

        # BATCH FEATURES
        batch_distance_features_matrixs = get_batch_distance_features_matrixs(docs)
        batch_grammar_features_matrixs = get_batch_grammar_features_matrixs(docs)
        batch_definiteness_features_matrixs = get_batch_definiteness_features_matrixs(docs)
        batch_match_features_scores = get_batch_match_features_scores(docs)
        batch_synonym_features_scores = get_batch_synonym_features_scores(docs)
        batch_hypernym_features_scores = get_batch_hypernym_features_scores(docs)

        # CONCATENATE ALL ANAPHOR SPAN REPRESENTATION WITH CANDIDATES SPAN REPRESENTATIONS AND CANDIDATES FEATURES
        for ana_id, ana_repr in enumerate(batch_anaphor_repr): # ana_repr: size [1, 256]
            doc_anaphor_potentials_concat = []
            potential_repr = batch_potential_repr[ana_id]
            # FEATURES PER ANAPHOR
            distance_features_matrixs = batch_distance_features_matrixs[ana_id]
            grammar_features_matrixs = batch_grammar_features_matrixs[ana_id]
            definiteness_features_matrixs = batch_definiteness_features_matrixs[ana_id]

            match_features_scores= batch_match_features_scores[ana_id]
            synonym_features_scores = batch_synonym_features_scores[ana_id]
            hypernym_features_scores = batch_hypernym_features_scores[ana_id]

            for pot_repr, pot_dis_feature_vec, pot_gramm_feature_vec, pot_definitness_feature_vec, match_features_score, synonym_features_score, hypernym_features_score in zip(potential_repr, distance_features_matrixs, grammar_features_matrixs, definiteness_features_matrixs, match_features_scores, synonym_features_scores, hypernym_features_scores):
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
                sigmoid_results_pro_sample.append(score) # different len because of different amount of potentials: list of potentials([512])
                labels_reults_pro_sample.append(assign_label(score))
            results_sigmoid.append(sigmoid_results_pro_sample)
            results_labels.append(labels_reults_pro_sample)
        return results_sigmoid, results_labels        #len = batch size : list of lists of potentials scores(tensors with one float e.g. tensor([0.27]))