from sklearn.utils import shuffle
from util import *
from loader import *
import torch.optim as optim
from torch import nn
from elmo_model_pairwise import Model
from bilstm_model import bilstm_model
from early_stopping import *
from transfer_learning_model_frozen import *


# EXPLAINATION OF THE 5-FOLD CROSSVALIDATION
'''
trainsets:  [2, 3, 4]  valset:  1  testset:  0
trainsets:  [1, 3, 4]  valset:  2  testset:  0
trainsets:  [1, 2, 4]  valset:  3  testset:  0
trainsets:  [1, 2, 3]  valset:  4  testset:  0

trainsets:  [2, 3, 4]  valset:  0  testset:  1
trainsets:  [0, 3, 4]  valset:  2  testset:  1
trainsets:  [0, 2, 4]  valset:  3  testset:  1
trainsets:  [0, 2, 3]  valset:  4  testset:  1

trainsets:  [1, 3, 4]  valset:  0  testset:  2
trainsets:  [0, 3, 4]  valset:  1  testset:  2
trainsets:  [0, 1, 4]  valset:  3  testset:  2
trainsets:  [0, 1, 3]  valset:  4  testset:  2

trainsets:  [1, 2, 4]  valset:  0  testset:  3
trainsets:  [0, 2, 4]  valset:  1  testset:  3
trainsets:  [0, 1, 4]  valset:  2  testset:  3
trainsets:  [0, 1, 2]  valset:  4  testset:  3

trainsets:  [1, 2, 3]  valset:  0  testset:  4
trainsets:  [0, 2, 3]  valset:  1  testset:  4
trainsets:  [0, 1, 3]  valset:  2  testset:  4
trainsets:  [0, 1, 2]  valset:  3  testset:  4'''


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
                 distance_feature=False,
                 grammar_role_feature=False,
                 definiteness_feature=False,
                 match_feature=False,
                 synonym_feature=False,
                 hypernym_feature=False,
                 new_model=False,
                 bert_model=False,
                 glove=False,
                 if_elmo=False,
                 if_bert=False,
                 device: torch.DeviceObjType = 'cuda' if torch.cuda.is_available() else 'cpu',
                 save_model=False,
                 transfer_learning=False):

        self.bert_model = bert_model
        self.train_feature = train_feature
        self.crossvalidation_sets = crossvalidation_sets
        self.early_stopping_patience = early_stopping_patience
        self.corpus_file_name = corpus_file_name
        self.k = folds_num
        self.save_model = save_model
        self.transfer_learning = transfer_learning

        self.distance_feature = distance_feature
        self.grammar_role_feature = grammar_role_feature
        self.definiteness_feature = definiteness_feature
        self.match_feature = match_feature
        self.synonym_feature = synonym_feature
        self.hypernym_feature = hypernym_feature


        self.glove = glove
        self.if_elmo = if_elmo
        self.if_bert = if_bert

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = None
        self.new_model=new_model

        self.lr = learning_rate
        self.device = device
        self.criterion = None
        self.optimizer = None

        self.seed = 20

        self.train_loss_array = []
        self.valid_loss_array = []
        self.eva_array = []

        # for evaluation
        self.num_all_correct = 0
        self.num_all_anaphor = 0
        self.all_sets_test_success_rate = 0

        # eva_file
        with open('../evaluation/{corpus_file_name}_{train_feature}_evaluation.txt'.format(corpus_file_name=self.corpus_file_name, train_feature=self.train_feature), 'w') as f:
            f.write('')

        # result_file
        with open('../results/{corpus_file_name}_{train_feature}_evaluation.txt'.format(corpus_file_name=self.corpus_file_name, train_feature=self.train_feature),'w') as f:
            f.write('')

    def _init_model(self):
        batch_size = 16  # the number of training examples utilized in one iteration = length of training samples
        if self.transfer_learning == False:
            if self.new_model:
                print('load new model')
                model = Model(batch_size=batch_size,
                            num_layers=1,
                            distance_feature=self.distance_feature,
                            grammar_role_feature=self.grammar_role_feature,
                            definiteness_feature=self.definiteness_feature,
                            match_feature=self.match_feature,
                            synonym_feature=self.synonym_feature,
                            hypernym_feature=self.hypernym_feature)
            else:
                # FOR OLD MODEL WITH EXTRA BILSTM
                print('load old bilstm model')
                model = bilstm_model(batch_size=batch_size,
                              num_layers=1,
                              bidirectional=True,
                              glove=self.glove,
                              if_elmo=self.if_elmo,
                              if_bert=self.if_bert,
                              distance_feature=self.distance_feature,
                              grammar_role_feature=self.grammar_role_feature,
                              definiteness_feature=self.definiteness_feature,
                              match_feature=self.match_feature,
                              synonym_feature=self.synonym_feature,
                              hypernym_feature=self.hypernym_feature)

    def train(self):
        # FOR EVERY DATASET/FOLD (5 FOLD => 5 DATASET)
        early_stopping = EarlyStopping(self.train_feature, corpus_name=self.corpus_file_name, patience=self.early_stopping_patience, pretrain=False, save_model=self.save_model)
        # FOR EVERY SET
        for cv_id, cv in enumerate(self.crossvalidation_sets):
            # initialize model for every Set
            self._init_model()
            self.model = self.model.to(self.device)
            self.criterion = nn.BCELoss().to(self.device)  # binary cross entropy
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            one_set_test_success_rate = 0
            with open('../results/{corpus_file_name}_{train_feature}_evaluation.txt'.format(corpus_file_name=self.corpus_file_name, train_feature=self.train_feature), 'a') as f:
                f.write('Cross Validation Set ' + str(cv_id+1) + '\n')

            Xy_train_sets = cv[0]
            Xy_val_sets = cv[1]
            Xy_test = cv[2]
            X_test = [xy[0] for xy in Xy_test]
            y_test = [xy[1] for xy in Xy_test]

            # FOR EVERY FOLD
            fold_id = 0
            for Xy_train, Xy_val in zip(Xy_train_sets, Xy_val_sets):
                early_stopping.fold_best_score = None

                with open('../results/{corpus_file_name}_{train_feature}_evaluation.txt'.format(corpus_file_name=self.corpus_file_name, train_feature=self.train_feature), 'a') as f:
                    f.write('Fold ' + str(fold_id + 1) + '\n')

                # INITIALIZE THE EARLY STOPPING OBJECT
                # for every cross-validation set
                X_train = [xy[0] for xy in Xy_train]
                y_train = [xy[1] for xy in Xy_train]

                X_val = [xy[0] for xy in Xy_val]
                y_val = [xy[1] for xy in Xy_val]

                fold_train_loss = 0.0
                fold_valid_loss = 0.0

                ### FOR EVER EPOCHE IN ONE FOLD
                for e in range(self.num_epochs):  # loop over the dataset multiple times
                    epoch_train_loss = 0.0
                    # Shuffle the corpus
                    X_train, y_train = shuffle(X_train, y_train, random_state=self.seed * e)
                    X_val, y_val = shuffle(X_val, y_val, random_state=self.seed * e)

                    ### TRAINING
                    ### EVERY BATCH
                    self.model.train()
                    for X_batch, y_batch in zip(batch(X_train, self.batch_size), batch(y_train, self.batch_size)):
                        self.optimizer.zero_grad()  # zero the parameter gradients
                        current_batch_size = len(X_batch)
                        if self.transfer_learning:
                            batch_concat_pairs, potential_slices = self.model(X_batch)
                        elif not self.new_model:
                            print('new_model: ', self.new_model)
                            batch_concat_pairs, potential_slices = self.model(X_batch, current_batch_size)  # SIGMOID RESULTS of one batch
                        else:
                            batch_concat_pairs, potential_slices = self.model(X_batch)  # SIGMOID RESULTS of one batch

                        batch_outputs, _ = self.model.predict(batch_concat_pairs, potential_slices)
                        flat_outputs = torch.cat([p for sample in batch_outputs for p in sample]).to(self.device)

                        flat_labels = torch.Tensor([p for sample in y_batch for p in sample]).to(self.device)

                        loss = self.criterion(flat_outputs, flat_labels)
                        # Backward pass
                        loss.backward()
                        self.optimizer.step()
                        epoch_train_loss += loss.item()
                        fold_train_loss += loss.item()

                    ### VALIDATION => validate every epoch to find and save the best model
                    epoch_valid_loss = 0.0
                    self.model.eval()
                    for X_batch, y_batch in zip(batch(X_val, self.batch_size), batch(y_val, self.batch_size)):
                        current_batch_size = len(X_batch)
                        # Forward Pass
                        if self.new_model:
                            batch_concat_pairs, potential_slices = self.model(X_batch)
                        else:
                            batch_concat_pairs, potential_slices = self.model(X_batch, current_batch_size)

                        batch_outputs, _ = self.model.predict(batch_concat_pairs, potential_slices)

                        flat_outputs = torch.cat([p for sample in batch_outputs for p in sample]).to(self.device)
                        flat_labels = torch.Tensor([l for sample in y_batch for l in sample]).to(self.device)
                        # Find the Loss
                        loss = self.criterion(flat_outputs, flat_labels)
                        # Calculate Loss
                        epoch_valid_loss += loss.item()
                        fold_valid_loss += loss.item()

                    self.train_loss_array.append(epoch_train_loss)
                    self.valid_loss_array.append(epoch_valid_loss)

                    # SAVE THE EPOCH LOSS EVERY EPOCHE
                    with open('../results/{corpus_file_name}_{train_feature}_evaluation.txt'.format(corpus_file_name=self.corpus_file_name, train_feature=self.train_feature), 'a') as f:
                        f.write('Epoch {e} \t\t Training Loss: {epoch_train_loss} \t\t Validation Loss: {epoch_valid_loss}\n\n'.format(e=e+1, epoch_train_loss=epoch_train_loss, epoch_valid_loss=epoch_valid_loss))

                    # update EARLY STOPPING
                    early_stopping(epoch_valid_loss, self.model, self.optimizer, e)
                    if early_stopping.early_stop_break_epoch == True:
                        with open('../results/{corpus_file_name}_{train_feature}_evaluation.txt'.format(corpus_file_name=self.corpus_file_name, train_feature=self.train_feature), 'a') as f:
                            f.write('Early stopped at Epoch {e} \n'.format(e=e))
                        early_stopping.early_stop_break_epoch = False
                        break

                # FOR EVERY FOLD
                # EVALUATE THE TEST SET
                one_fold_test_success_rate = self.evaluate(X_test, y_test, fold_id)
                one_set_test_success_rate += one_fold_test_success_rate
                with open('../results/{corpus_file_name}_{train_feature}_evaluation.txt'.format(corpus_file_name=self.corpus_file_name, train_feature=self.train_feature), 'a') as f:
                    f.write('Set {cv_id} Fold {fold_id} test set success rate: {one_fold_test_success_rate}\n\n'.format(
                            cv_id=cv_id + 1, fold_id=fold_id + 1, one_fold_test_success_rate=one_fold_test_success_rate))
                fold_id += 1

            average_one_set_test_success_rate = one_set_test_success_rate/(self.k-1)
            with open('../results/{corpus_file_name}_{train_feature}_evaluation.txt'.format(corpus_file_name=self.corpus_file_name, train_feature=self.train_feature), 'a') as f:
                f.write('Set {cv_id} average test set success rate: {average_one_set_test_success_rate}\n\n'.format(
                    cv_id=cv_id + 1, average_one_set_test_success_rate=average_one_set_test_success_rate))

            self.all_sets_test_success_rate += average_one_set_test_success_rate

            cv_id += 1

        average_test_success_rate_by_counting = self.num_all_correct/self.num_all_anaphor
        average_test_success_rate_by_averaging = self.all_sets_test_success_rate/self.k

        with open('../results/{corpus_file_name}_{train_feature}_evaluation.txt'.format(corpus_file_name=self.corpus_file_name, train_feature=self.train_feature), 'a') as f:
            f.write('\n\n Final average TEST success rate by counting: {average_test_success_rate_by_counting}'.format(average_test_success_rate_by_counting=average_test_success_rate_by_counting))
            f.write('\n\n Final average TEST success rate by averaging: {average_test_success_rate_by_averaging}'.format(average_test_success_rate_by_averaging=average_test_success_rate_by_averaging))
            f.write('\n\n Number of all anaphors: {num_samp} \n\n Number of all correct predicted anaphors: {num_correct}'.format(num_samp=self.num_all_anaphor, num_correct=self.num_all_correct))

        with open("./arrays/{corpus_name}_{train_feature}_train_loss.txt".format(corpus_name=self.corpus_file_name,train_feature=self.train_feature), "w") as fp:
            fp.write(json.dumps(self.train_loss_array))

        with open("./arrays/{corpus_name}_{train_feature}_valid_loss.txt".format(corpus_name=self.corpus_file_name,train_feature=self.train_feature), "w") as fp:
            fp.write(json.dumps(self.valid_loss_array))

        with open("./arrays/{corpus_name}_{train_feature}_eva.txt".format(corpus_name=self.corpus_file_name,train_feature=self.train_feature), "w") as fp:
            fp.write(json.dumps(self.eva_array))

    # EVALUATE the model
    def evaluate(self, X_test, y_test, fold_id):
        '''
        One evaluation per cross validation set.
        Evaluate the model on the test set.
        @return: Success rate = successfully resolved anaphors/number of all anaphors
        '''
        if fold_id == 3:
            eva_file = open('../evaluation/{corpus_file_name}_{train_feature}_evaluation.txt'.format(corpus_file_name=self.corpus_file_name, train_feature=self.train_feature), 'a')
        self.model.eval()
        correct = 0
        num_anaphors = 0

        for X_batch, y_batch in zip(batch(X_test, self.batch_size), batch(y_test, self.batch_size)):
            current_batch_size = len(X_batch)
            if not self.new_model:
                batch_concat_pairs, potential_slices = self.model(X_batch, current_batch_size)  # SIGMOID RESULTS of one batch
            else:
                batch_concat_pairs, potential_slices = self.model(X_batch)  # SIGMOID RESULTS of one batch
            batch_label_outputs, _ = self.model.predict(batch_concat_pairs, potential_slices)

            ana_strs = [ana["tokens"] for ana in X_batch]
            ana_contexts = [ana['context'] for ana in X_batch]
            gold_strs = [ana["gold_str"] for ana in X_batch]
            all_ps_strs = [[p["tokens"] for p in ana["potential_antecedents"]] for ana in X_batch]

            # EVERY SAMPLE
            i = 0
            for targets, ys in zip(batch_label_outputs, y_batch):
                num_anaphors += 1
                if fold_id == 3:
                    eva_file.write('anaphor: ' + str(ana_strs[i]) + '\n')
                    eva_file.write('candidates: ' + str(all_ps_strs[i]) + '\n')
                    eva_file.write('gold: ' + str(gold_strs[i]) + '\n')
                # PREDICTION: find the index of the max value of sigmoid?
                selected_index = targets.index(max(targets))
                if fold_id == 3:
                    eva_file.write('selected cand: ' + str(all_ps_strs[i][selected_index]) + '\n')
                    eva_file.write('context: ' + str(ana_contexts[i]) + '\n')

                # if one of the candidates is gold
                if 1 in ys: # because its possible that none of the potential candicates is right(e.g. the gold antecedent is not in the 3 sents before)
                    gold_and_coref_idxs = [i for i, val in enumerate(ys) if val == 1]
                    if selected_index in gold_and_coref_idxs:
                        correct += 1
                        if fold_id == 3:
                            eva_file.write('right\n')
                    else:
                        if fold_id == 3:
                            eva_file.write('wrong\n')
                else:
                    if fold_id == 3:
                        eva_file.write('wrong\n')
                if fold_id == 3:
                    eva_file.write('\n')
                i += 1

        self.num_all_correct += correct
        self.num_all_anaphor += num_anaphors
        success_rate = correct / num_anaphors

        if fold_id == 3:
            eva_file.close()

        return success_rate


if __name__ == '__main__':
    corpus_names = ['anaphor_with', 'anaphor_without']
    i = 1

    # Train 3 corpus
    for corpus_name in corpus_names:
        train_feature = 'bert_glove_elmo' #   2_embeddings_6_features   |   2_embeddings   |   only_elmo   |   2_emb_only_distance_fea

        early_stopping_patience = 8
        num_epochs = 100  # klowersa = 20
        batch_size = 16
        learning_rate = 1e-5
        folds_num = 5
        crossvalidation_sets = load_corpus_list(corpus_name)

        trainer = Trainer(train_feature,
                         early_stopping_patience,
                         learning_rate,
                         num_epochs,
                         batch_size,
                         corpus_name,
                         folds_num,
                         crossvalidation_sets,
                         distance_feature=False,
                         grammar_role_feature=False,
                         definiteness_feature=False,
                         match_feature=False,
                         synonym_feature=False,
                         hypernym_feature=False,

                         new_model=True,
                         if_bert=True,

                         glove=False,
                         if_elmo=False,
                         save_model=False,
                         transfer_learning=False)

        trainer.train()
        i += 1