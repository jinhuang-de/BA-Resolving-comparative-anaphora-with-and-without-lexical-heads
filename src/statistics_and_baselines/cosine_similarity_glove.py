from loader import *
from sklearn.utils import shuffle

import torch.optim as optim
from torch import nn
import numpy as np
import torch
from typing import *
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence
import gensim

class bilstm_model(nn.Module):
    def __init__(self,
                 batch_size: int,
                 train_feature,
                 num_layers: int = 1,
                 bidirectional: bool = True,
                 glove=True,
                 if_elmo=False,
                 pooling=False,
                 average_after_pooling=False,
                 only_head=False,
                 device: torch.DeviceObjType = 'cuda' if torch.cuda.is_available() else 'cpu'):  # input_size: int => ocab_size

        # avg + noPooling
        # avg + Pooling
        # ! sum + noPooling
        # sum + Pooling
        super(bilstm_model, self).__init__()

        self.cosine_path = "../results/cosine_similarity_{train_feature}.txt".format(train_feature=train_feature)

        # cosine similary

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.all_cosine_similarity = torch.Tensor([0.])
        self.num_of_all_ana = 0
        self.all_non_gold_candidate_with_anaphor_similarity = 0
        self.all_non_gold_candidate = 0
        self.all_gold_and_coref_with_anaphor_similarity = 0
        self.all_gold_and_coref = 0

        # evaluation
        self.batch_correct = None
        self.batch_all_ana = None

        self.whole_corpus = Corpus().corpus_total
        self.device = device
        self.glove = glove

        self.if_elmo = if_elmo

        self.pooling = pooling

        self.useless_gold = 0
        self.average_after_pooling = average_after_pooling
        self.only_head = only_head

        # SIZES
        self.batch_size = batch_size
        self.current_batch_size = batch_size

        self.num_layers = num_layers

        self.current_batch_size = batch_size

        self.max_pooling = nn.MaxPool1d(kernel_size=2)

        self.glove_word_to_idx = None
        self.glove_weight = None
        self.glove_embeddings = None

        if self.glove:
            self.glove_weight = self.get_glove_weight()
            self.glove_embeddings = nn.Embedding.from_pretrained(self.glove_weight).to(self.device)
            self.glove_embeddings.weight.requires_grad = False

        self.deleted_sample = 0

    def _init_hidden(self, current_batch_size) -> Tuple[torch.Tensor]:
        '''Initialize the hidden layers in forward function
        @param current_batch_size: the current batch size(Every batch size is diffrent because every anaphor has different amount of candidates.)
        '''
        h = c = torch.zeros(self.num_layers * self.directions, current_batch_size, self.bilstm_hidden_size)
        self.current_batch_size = current_batch_size
        return h.to(self.device), c.to(self.device)

    # TODO: ANPASSEN FÜR NEUE DATEN
    def get_idxs(self, batch_doc):
        '''Get the idx of the anaphor and its candidates antecedents in its context.
        @param doc: a batch of anaphor objects
        @return: 2 lists --- the anaphor slices and slices of its candidates antecedents of a batch
        '''
        anaphor_slices = []
        potential_slices = []
        gold_slices = []

        for doc in batch_doc:
            mention = doc
            first_index_of_context = mention["first_index_of_context"]
            if self.only_head == False:
                anaphor_slice = [mention["left"] - first_index_of_context, mention["right"] - first_index_of_context]
                potential_slice = [[p["left"] - first_index_of_context, p["right"] - first_index_of_context] for p in mention["potential_antecedents"]]
                gold_slice = [mention["gold"]["left"] - first_index_of_context, mention["gold"]["right"] - first_index_of_context]

            else:
                anaphor_slice = [mention["head_left"] - first_index_of_context, mention["head_right"] - first_index_of_context]
                potential_slice = [[p["head_left"] - first_index_of_context, p["head_right"] - first_index_of_context] for p in mention["potential_antecedents"]]
                gold_slice = [mention["gold"]["head_left"] - first_index_of_context, mention["gold"]["head_right"] - first_index_of_context]

            anaphor_slices.append(anaphor_slice)
            potential_slices.append(potential_slice)
            gold_slices.append(gold_slice)

        return anaphor_slices, potential_slices, gold_slices

    def get_batch_neural_span_of_anaphor(self, batch_lstm_out: torch.Tensor, idx: List):  # lstm output 16 x seq length x 512
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

    def get_batch_neural_span_of_gold(self, batch_lstm_out: torch.Tensor,
                                      idx: List):  # lstm output 16 x seq length x 512
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

    def get_batch_neural_span_of_potentials(self, batch_lstm_out: torch.Tensor,
                                            potentials_idx: List):  # lstm output 16 x seq length x 512
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
        batch_glove_idxs_tensor = pad_sequence(batch_glove_idxs, batch_first=True, padding_value=1)  # because the index of <pad> is 1

        batch_glove_embeddings = []
        for samp_idx in batch_glove_idxs_tensor:
            sample_idxs = []
            # embed every word
            for word_idx in samp_idx:
                word_embedding = self.glove_embeddings(word_idx.to(self.device)).view(-1).to(self.device)  # [300]
                sample_idxs.append(word_embedding)
            batch_glove_embeddings.append(torch.stack(sample_idxs))

        batch_glove_embeddings = torch.stack(batch_glove_embeddings)  # [16, 97, 300] [batch, max_seq, glove_emb_dim]

        return batch_glove_embeddings.to(self.device)

    def delete_useless_samples(self, batch_anaphor_repr, batch_gold_repr, batch_potential_repr):
      '''
      Delete samples, whose gold label is not in context.
      '''
      idxs = []
      idx = 0
      for g_repr in batch_gold_repr:
          if g_repr.size()[0] == 0:
             self.deleted_sample += 1
             idxs.append(idx)
          idx += 1

      batch_anaphor_repr = list(np.delete(batch_anaphor_repr, idxs))
      batch_gold_repr = list(np.delete(batch_gold_repr, idxs))
      batch_potential_repr = list(np.delete(batch_potential_repr, idxs))

      return batch_anaphor_repr, batch_gold_repr, batch_potential_repr

    def forward(self, docs, current_batch_size) -> List[torch.Tensor]:
        '''
        @param docs: a BACTH of anaphor object
        @param current_batch_size: the current batch size(Every batch size is diffrent because every anaphor has different amount of candidates.)
        @return:
        '''
        all_batch_correct = 0
        all_batch_ana = 0
        embeddings = self.get_batch_glove_embeddings(docs)

        # For every batch in last_hidden
        # Find the Slices of spans mentions in thier context
        anaphor_slices, potential_slices, gold_slices = self.get_idxs(docs)

        last_hidden = embeddings

        # Extract their span representations from the hidden matrix
        batch_anaphor_repr = self.get_batch_neural_span_of_anaphor(last_hidden, anaphor_slices)
        batch_potential_repr = self.get_batch_neural_span_of_potentials(last_hidden, potential_slices)
        batch_gold_repr = self.get_batch_neural_span_of_gold(last_hidden, gold_slices)
        # for g in batch_gold_repr:
        #     if torch.isnan(g).any():
        #         print('original gold: ', g)

        # for g in batch_gold_repr:
        #     if len(g.size()) == torch.Size([0, 300]):
        #         print('original gold: ', g)

        # batch_anaphor_repr, batch_gold_repr, batch_potential_repr = self.delete_useless_samples(batch_anaphor_repr, batch_gold_repr, batch_potential_repr)

        # with maxpooling
        if self.pooling == True:
            # for ana_repr in batch_anaphor_repr:
            #     print(ana_repr.size())
            batch_anaphor_repr = [self.max_pooling(ana_repr) for ana_repr in batch_anaphor_repr]
            batch_gold_repr = [self.max_pooling(gold_repr) for gold_repr in batch_gold_repr]

        # + sum
        if self.average_after_pooling == False:
            batch_anaphor_repr = [torch.sum(ana_repr, 0) for ana_repr in batch_anaphor_repr]
            batch_gold_repr = [torch.sum(gold_repr, 0) for gold_repr in batch_gold_repr]
        # + avg
        else:
            batch_anaphor_repr = [torch.mean(ana_repr, 0) for ana_repr in batch_anaphor_repr]
            batch_gold_repr = [torch.mean(gold_repr, 0) for gold_repr in batch_gold_repr]

        # TODO: gold: all NAN
        # for g in batch_gold_repr:
        #     if torch.isnan(g).any():
        #         print('pooled + avg gold: ', g)

        max_pooled_batch_potential_repr = []
        # for all cands of one anaphor
        for sample in batch_potential_repr:  # batch_potential_repr: len = batch_size
            p_reprs_pro_sample = []
            # for each cand
            for p_repr in sample:
                new_p_repr = None
                # if pooling
                if self.pooling == True:
                    new_p_repr = self.max_pooling(p_repr)  # before sum [x, 1, 256]
                if new_p_repr == None:
                    if self.average_after_pooling == True:
                        new_p_repr = torch.mean(p_repr, 0)
                    else:
                        new_p_repr = torch.sum(p_repr, 0)
                else:
                    if self.average_after_pooling == True:
                        new_p_repr = torch.mean(new_p_repr, 0)
                    else:
                        new_p_repr = torch.sum(new_p_repr, 0)

                p_reprs_pro_sample.append(new_p_repr)
            max_pooled_batch_potential_repr.append(p_reprs_pro_sample)
        batch_potential_repr = max_pooled_batch_potential_repr

        ana_strs = [ana["tokens"] for ana in docs]
        gold_strs = [ana["gold_str"] for ana in docs]
        docs_pots_strs = [[p["tokens"] for p in ana["potential_antecedents"]] for ana in docs]
        golds = [ana['gold'] for ana in docs]
        docs_ps_docs = [[p for p in ana["potential_antecedents"]] for ana in docs]

        for i in range(len(batch_anaphor_repr)):
            all_batch_ana += 1
            ana_r = batch_anaphor_repr[i]
            gold_r = batch_gold_repr[i]
            batch_potential_reprs = batch_potential_repr[i]
            ana_str = ana_strs[i]
            gold_str = gold_strs[i]
            pots_strs = docs_pots_strs[i]
            ps_docs = docs_ps_docs[i]
            # if not isinstance(ps_docs, List):
            #     ps_docs = []
            gold = golds[i]
            ana = docs[i]

            gold_corefs = [gold["id"]]
            if ana["corefs_ids"] != None:
                gold_corefs = gold_corefs + ana["corefs_ids"]

            print('context: ', docs[i]['context'], '\n', file=open(self.cosine_path, "a"))
            print('anaphor: ', ana_str, '\n', file=open(self.cosine_path, "a"))

            # nn.cosine_similarity need 1+ dimensions
            cos_similarity = self.cos(ana_r.unsqueeze(0), gold_r.unsqueeze(0)).item()

            # if torch.isnan(self.cos(ana_r.unsqueeze(0), gold_r.unsqueeze(0))):
            #     print('ana: ', ana_r.unsqueeze(0))
            #     print('gold: ', gold_r.unsqueeze(0))

            print('gold: ', gold_str, ' cos_similarity to anaphor: ', cos_similarity, '\n', file=open(self.cosine_path, "a"))

            candidates_labels_for_one_ana = []
            candidates_cosines = []
            # for each candidate of an anaphor
            for p_str, p_repr, p in zip(pots_strs, batch_potential_reprs, ps_docs):
                p_corefs = [p['id']]
                if p['corefs'] != None:
                    p_corefs = p['corefs'] + p_corefs
                # cosine similarity between anaphor and candidate
                cos_similarity = self.cos(ana_r.unsqueeze(0), p_repr.unsqueeze(0)).item()

                # if torch.isnan(self.cos(ana_r.unsqueeze(0), p_repr.unsqueeze(0))):
                #     print('p_repr: ', p_repr.unsqueeze(0))

                candidates_cosines.append(cos_similarity)

                if any(x in p_corefs for x in gold_corefs):
                    print('coref candidate: ', p_str, ' cos_similarity to ana: ', cos_similarity, '\n', file=open(self.cosine_path, "a"))
                    self.all_gold_and_coref_with_anaphor_similarity += cos_similarity
                    self.all_gold_and_coref += 1
                    candidates_labels_for_one_ana.append(1)

                else:
                    print('candidate: ', p_str, ' cos_similarity to ana: ', cos_similarity, '\n', file=open(self.cosine_path, "a"))
                    self.all_non_gold_candidate_with_anaphor_similarity += cos_similarity
                    self.all_non_gold_candidate += 1
                    candidates_labels_for_one_ana.append(0)

            max_index = candidates_cosines.index(max(candidates_cosines))
            if candidates_labels_for_one_ana[max_index] == 1:
                all_batch_correct += 1

            print('\n\n', file=open(self.cosine_path, "a"))

        return all_batch_ana, all_batch_correct


class Trainer:
    def __init__(self,
                 train_feature,
                 early_stopping_patience,
                 learning_rate,
                 num_epochs,
                 batch_size,
                 corpus_file_name,
                 folds_num,
                 crossvalidation_sets,
                 pooling,
                 average_after_pooling,
                 new_model=True,
                 glove=False,
                 if_elmo=False,
                 device: torch.DeviceObjType = 'cuda' if torch.cuda.is_available() else 'cpu',
                 save_model=False,
                 transfer_learning=False):
        '''
        @param model:
        '''

        self.pooling = pooling
        self.average_after_pooling = average_after_pooling

        self.train_feature = train_feature
        self.crossvalidation_sets = crossvalidation_sets
        self.early_stopping_patience = early_stopping_patience
        self.corpus_file_name = corpus_file_name
        self.k = folds_num
        self.save_model = save_model
        self.transfer_learning = transfer_learning

        self.glove = glove
        self.if_elmo = if_elmo

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = None
        self.new_model = new_model

        self.lr = learning_rate
        self.device = device

        self.seed = 20

        self.train_loss_array = []
        self.valid_loss_array = []
        self.eva_array = []

        # for evaluation
        self.num_all_correct = 0
        self.num_all_anaphor = 0
        self.all_sets_test_success_rate = 0

        self.crossvalidation_sets = crossvalidation_sets

    def _init_model(self):

        # FOR OLD MODEL WITH EXTRA BILSTM
        model = bilstm_model(batch_size=batch_size,
                             train_feature=train_feature,
                             num_layers=1,
                             bidirectional=True,
                             glove=self.glove,
                             if_elmo=self.if_elmo,
                             pooling = self.pooling,
                             average_after_pooling = self.average_after_pooling)


        self.model = model.to(self.device)

    def train(self):
        ### FOR EVER EPOCHE

        self._init_model()

        cv = self.crossvalidation_sets[0]

        Xy_train_sets = cv[0]
        Xy_val_sets = cv[1]
        Xy_test = cv[2]

        X_train = [xy[0] for xy in Xy_train_sets[0]]
        y_train = [xy[1] for xy in Xy_train_sets[0]]

        X_val = [xy[0] for xy in Xy_val_sets[0]]
        y_val = [xy[1] for xy in Xy_val_sets[0]]

        X_test = [xy[0] for xy in Xy_test]
        y_test = [xy[1] for xy in Xy_test]

        X_train = X_train + X_val + X_test
        y_train = y_train + y_val + y_test

        # Shuffle the corpus
        X_train, y_train = shuffle(X_train, y_train, random_state=self.seed)

        ### TRAINING
        ### EVERY BATCH
        for X_batch, y_batch in zip(batch(X_train, self.batch_size), batch(y_train, self.batch_size)):
            current_batch_size = len(X_batch)

            all_batch_ana, all_batch_correct = self.model(X_batch, current_batch_size)  # SIGMOID RESULTS of one batch
            self.num_all_correct += all_batch_correct
            self.num_all_anaphor += all_batch_ana
        print('deleted ', self.model.deleted_sample, ' samples.', file=open(self.model.cosine_path, "a"))

        print('average non gold candidates with anaphor similarity: ',
              self.model.all_non_gold_candidate_with_anaphor_similarity / self.model.all_non_gold_candidate,
              file=open(self.model.cosine_path, "a"))
        print('average gold and coref with anaphor similarity: ',
              self.model.all_gold_and_coref_with_anaphor_similarity / self.model.all_gold_and_coref,
              file=open(self.model.cosine_path, "a"))
        print('glove accuracy: ', self.num_all_correct / self.num_all_anaphor, file=open(self.model.cosine_path, "a"))
        print('\n\n\n\n', file=open(self.model.cosine_path, "a"))

if __name__ == '__main__':
    corpus_names = ['anaphor_without']
    i = 1

    # Train 3 corpus
    for corpus_name in corpus_names:
        early_stopping_patience = 8
        num_epochs = 100  # klowersa = 20
        batch_size = 16
        learning_rate = 1e-6
        folds_num = 5
        crossvalidation_sets = load_corpus_list(corpus_name)


        train_feature = 'without_bilstm_glove_sum_noPooling'  # 2_embeddings_6_features   |   2_embeddings   |   only_elmo   |   2_emb_only_distance_fea
        trainer = Trainer(train_feature,
                          early_stopping_patience,
                          learning_rate,
                          num_epochs,
                          batch_size,
                          corpus_name,
                          folds_num,
                          crossvalidation_sets,
                          pooling=False,
                          average_after_pooling=False,
                          new_model=False,

                          glove=True,
                          if_elmo=False,
                          save_model=False,
                          transfer_learning=False)
        trainer.train()

        train_feature = 'without_bilstm_glove_sum_Pooling'  # 2_embeddings_6_features   |   2_embeddings   |   only_elmo   |   2_emb_only_distance_fea
        trainer = Trainer(train_feature,
                          early_stopping_patience,
                          learning_rate,
                          num_epochs,
                          batch_size,
                          corpus_name,
                          folds_num,
                          crossvalidation_sets,
                          pooling=True,
                          average_after_pooling=False,
                          new_model=False,

                          glove=True,
                          if_elmo=False,
                          save_model=False,
                          transfer_learning=False)
        trainer.train()


        i += 1