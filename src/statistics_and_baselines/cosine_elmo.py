from typing import *
import torch.nn as nn
import allennlp.modules.elmo as E
from loader import *
from sklearn.utils import shuffle
from util import *
import torch.optim as optim

from early_stopping import *


class Model(nn.Module):
    def __init__(self,
                 batch_size: int,
                 train_feature,
                 num_layers: int = 1,
                 pooling=False,
                 average_after_pooling=False,
                 only_head=False,
                 device: torch.DeviceObjType = 'cuda' if torch.cuda.is_available() else 'cpu'):  # input_size: int => ocab_size

        super(Model, self).__init__()
        self.cosine_path = "../results/cosine_similarity_{train_feature}.txt".format(train_feature=train_feature)

        self.all_non_gold_candidate_with_anaphor_similarity = 0
        self.all_non_gold_candidate = 0
        self.all_gold_and_coref_with_anaphor_similarity = 0
        self.all_gold_and_coref = 0

        self.whole_corpus = Corpus().corpus_total
        self.device = device

        # evaluation
        self.batch_correct = None
        self.batch_all_ana = None

        self.pooling = pooling

        self.only_head = only_head
        self.useless_gold = 0
        self.average_after_pooling = average_after_pooling

        self.current_batch_size = batch_size

        # cosine similary
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.all_cosine_similarity = torch.Tensor([0.])
        self.num_of_all_ana = 0

        self.max_pooling = nn.MaxPool1d(kernel_size=2).to(self.device)
        # EMBEDDINGS
        elmo_options_file = '../embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
        elmo_weight_file = '../embeddings/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'

        self.elmo = E.Elmo(elmo_options_file, elmo_weight_file, num_output_representations=1, requires_grad=True).to(
            self.device)  # , dropout=0.1

        self.deleted_sample = 0

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
        gold_slices = []

        for doc in batch_doc:
            mention = doc
            first_index_of_context = mention["first_index_of_context"]
            if self.only_head == False:
                anaphor_slice = [mention["left"] - first_index_of_context, mention["right"] - first_index_of_context]
                potential_slice = [[p["left"] - first_index_of_context, p["right"] - first_index_of_context] for p in
                                   mention["potential_antecedents"]]
                gold_slice = [mention["gold"]["left"] - first_index_of_context,
                              mention["gold"]["right"] - first_index_of_context]

            else:
                anaphor_slice = [mention["head_left"] - first_index_of_context,
                                 mention["head_right"] - first_index_of_context]
                potential_slice = [[p["head_left"] - first_index_of_context, p["head_right"] - first_index_of_context]
                                   for p in mention["potential_antecedents"]]
                gold_slice = [mention["gold"]["head_left"] - first_index_of_context,
                              mention["gold"]["head_right"] - first_index_of_context]

            anaphor_slices.append(anaphor_slice)
            potential_slices.append(potential_slice)
            gold_slices.append(gold_slice)

        return anaphor_slices, potential_slices, gold_slices

    def get_batch_neural_span_of_anaphor(self, batch_lstm_out: torch.Tensor,
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

        elmo_embeddings = self.get_context_elmo_embeddings(
            docs)  # (batch_size, mex_sample_len, 256)  e.g. [16, 64, 256]
        last_hidden = elmo_embeddings

        anaphor_slices, potential_slices, gold_slices = self.get_idxs(docs)

        # Extract their span representations from the hidden matrix
        batch_anaphor_repr = self.get_batch_neural_span_of_anaphor(last_hidden, anaphor_slices)
        batch_potential_repr = self.get_batch_neural_span_of_potentials(last_hidden, potential_slices)
        batch_gold_repr = self.get_batch_neural_span_of_gold(last_hidden, gold_slices)

        batch_anaphor_repr, batch_gold_repr, batch_potential_repr = self.delete_useless_samples(batch_anaphor_repr,
                                                                                     batch_gold_repr,                                                                                        batch_potential_repr)
        # with maxpooling
        if self.pooling == True:
            batch_anaphor_repr = [self.max_pooling(ana_repr.unsqueeze(0)).squeeze(0) for ana_repr in batch_anaphor_repr]
            batch_gold_repr = [self.max_pooling(gold_repr.unsqueeze(0)).squeeze(0) for gold_repr in batch_gold_repr]

        # + sum
        if self.average_after_pooling == False:
            batch_anaphor_repr = [torch.sum(ana_repr, 0) for ana_repr in batch_anaphor_repr]
            batch_gold_repr = [torch.sum(gold_repr, 0) for gold_repr in batch_gold_repr]

        # + avg
        else:
            batch_anaphor_repr = [torch.mean(ana_repr, 0) for ana_repr in batch_anaphor_repr]
            batch_gold_repr = [torch.mean(gold_repr, 0) for gold_repr in batch_gold_repr]

        max_pooled_batch_potential_repr = []
        # for all cands of one anaphor
        for sample in batch_potential_repr:  # batch_potential_repr: len = batch_size
            p_reprs_pro_sample = []
            # for each cand
            for p_repr in sample:
                new_p_repr = None
                # if pooling
                if self.pooling == True:
                    new_p_repr = self.max_pooling(p_repr.unsqueeze(0)).squeeze(0)  # before sum [x, 1, 256]
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
            gold = golds[i]
            ana = docs[i]
            gold_corefs = [gold["id"]]
            if ana["corefs_ids"] != None:
                gold_corefs = gold_corefs + ana["corefs_ids"]

            print('context: ', docs[i]['context'], '\n', file=open(self.cosine_path, "a"))
            print('anaphor: ', ana_str, '\n', file=open(self.cosine_path, "a"))

            # nn.cosine_similarity need 1+ dimensions
            cos_similarity = self.cos(ana_r.unsqueeze(0), gold_r.unsqueeze(0)).item()
            print('gold: ', gold_str, ' cos_similarity to anaphor: ', cos_similarity, '\n',
                  file=open(self.cosine_path, "a"))

            candidates_labels_for_one_ana = []
            candidates_cosines = []
            # for each candidate of an anaphor
            for p_str, p_repr, p in zip(pots_strs, batch_potential_reprs, ps_docs):
                p_corefs = [p['id']]
                if p['corefs'] != None:
                    p_corefs = p['corefs'] + p_corefs
                # cosine similarity between anaphor and candidate
                cos_similarity = self.cos(ana_r.unsqueeze(0), p_repr.unsqueeze(0)).item()
                candidates_cosines.append(cos_similarity)

                if any(x in p_corefs for x in gold_corefs):
                    print('coref candidate: ', p_str, ' cos_similarity to ana: ', cos_similarity, '\n',
                          file=open(self.cosine_path, "a"))
                    self.all_gold_and_coref_with_anaphor_similarity += cos_similarity
                    self.all_gold_and_coref += 1
                    candidates_labels_for_one_ana.append(1)

                else:
                    print('candidate: ', p_str, ' cos_similarity to ana: ', cos_similarity, '\n',
                          file=open(self.cosine_path, "a"))
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
        self.corpus_file_name = corpus_file_name

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = None
        self.new_model = new_model

        self.device = device

        self.seed = 20

        self.train_loss_array = []
        self.valid_loss_array = []
        self.eva_array = []

        # for evaluation
        self.num_all_correct = 0
        self.num_all_anaphor = 0
        self.all_sets_test_success_rate = 0

    def _init_model(self):
        model = Model(batch_size=batch_size,
                      train_feature=self.train_feature,
                      num_layers=1,
                      bidirectional=True,
                      glove=self.glove,
                      if_elmo=self.if_elmo,
                      pooling=self.pooling,
                      average_after_pooling=self.average_after_pooling)

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
        print('elmo accuracy: ', self.num_all_correct / self.num_all_anaphor, file=open(self.model.cosine_path, "a"))
        print('\n\n\n\n', file=open(self.model.cosine_path, "a"))


if __name__ == '__main__':
    corpus_names = ['anaphor_without']
    i = 1

    # Train 3 corpus
    for corpus_name in corpus_names:
        early_stopping_patience = 8
        num_epochs = 100  # klowersa = 20
        batch_size = 16
        learning_rate = 1e-5
        folds_num = 5
        crossvalidation_sets = load_corpus_list(corpus_name)



        train_feature = 'elmo_sum_noPooling'  # 2_embeddings_6_features   |   2_embeddings   |   only_elmo   |   2_emb_only_distance_fea
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

