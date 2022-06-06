from sklearn.utils import shuffle
from ontonotes_utils import *
from loader_ontonotes_pronoun import *
import torch.optim as optim
from torch import nn
from pretrain_elmo_model import Model
import numpy as np
from early_stopping import *
import torch

class Trainer:
    def __init__(self,
                 train_feature,
                 model,
                 early_stopping_patience,
                 learning_rate,
                 num_epochs,
                 batch_size,
                 X_train,
                 X_val,
                 X_test,
                 y_train,
                 y_val,
                 y_test,
                 device: torch.DeviceObjType = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.train_feature = train_feature
        self.early_stopping_patience = early_stopping_patience
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.model = model.to(self.device)
        self.lr = learning_rate
        self.criterion = nn.BCELoss().to(self.device)  # binary cross entropy
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.seed = 20

        self.train_loss_path = "../arrays/{train_feature}_train_loss.txt".format(train_feature=self.train_feature)
        self.valid_loss_path = "../arrays/{train_feature}_valid_loss.txt".format(train_feature=self.train_feature)
        self.eva_path = "../arrays/pretrain_{train_feature}_eva.txt".format(train_feature=self.train_feature)

        self.train_loss_array = []
        self.valid_loss_array = []

        self.success_rate = None

        # for evaluation
        self.num_all_correct = 0
        self.num_all_anaphor = 0

        # eva_file
        with open('../evaluation/{train_feature}_evaluation.txt'.format(train_feature=self.train_feature),
                  'w') as f:
            f.write('')

        # results_file
        with open('../results/pretrain_{train_feature}_evaluation.txt'.format(train_feature=self.train_feature),
                  'w') as f:
            f.write('')

        # losses file
        with open(self.train_loss_path, "w") as fp:
            fp.write(json.dumps(self.train_loss_array))

        with open(self.valid_loss_path, "w") as fp:
            fp.write(json.dumps(self.valid_loss_array))

        # new losses file
        self.avg_train_loss_path = "../arrays/avg_{train_feature}_train_loss.txt".format(train_feature=self.train_feature)
        self.avg_valid_loss_path = "../arrays/avg_{train_feature}_valid_loss.txt".format(train_feature=self.train_feature)
        self.avg_train_loss_array = []
        self.avg_valid_loss_array = []
        with open(self.avg_train_loss_path, "w") as fp:
            fp.write(json.dumps(self.avg_train_loss_array))
        with open(self.avg_valid_loss_path, "w") as fp:
            fp.write(json.dumps(self.avg_valid_loss_array))

    def train(self):
        sum_train_loss = 0
        sum_val_loss = 0

        # INITIALIZE THE EARLY STOPPING OBJECT
        early_stopping = EarlyStopping(self.train_feature, patience=self.early_stopping_patience, pretrain=True, save_model=True)

        ### FOR EVER EPOCHE
        for e in range(self.num_epochs):  # loop over the dataset multiple times
            epoch_train_sample_pairs_len = 0
            epoch_val_sample_pairs_len = 0
            epoch_train_loss = 0.0

            # Shuffle the corpus
            X_train, y_train = shuffle(self.X_train, self.y_train, random_state=self.seed * e)
            X_val, y_val = shuffle(self.X_val, self.y_val, random_state=self.seed * e)

            ### TRAINING
            ### EVERY BATCH
            self.model.train()
            for X_batch, y_batch in zip(batch(X_train, self.batch_size), batch(y_train, self.batch_size)):
                self.optimizer.zero_grad()  # zero the parameter gradients
                current_batch_size = len(X_batch)

                batch_concat_pairs, potential_slices = self.model(X_batch)  # SIGMOID RESULTS of one batch
                batch_outputs, _ = self.model.predict(batch_concat_pairs, potential_slices)

                for sample_results in batch_outputs:
                    epoch_train_sample_pairs_len += len(sample_results)
                # if torch.cuda.is_available():
                flat_outputs = torch.cat([p for sample in batch_outputs for p in sample]).to(self.device)
                flat_labels = torch.Tensor([l for sample in y_batch for l in sample]).to(self.device)

                loss = self.criterion(flat_outputs, flat_labels)
                # Backward pass
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
                sum_train_loss += loss.item()

                with open(self.train_loss_path, 'r') as f:
                    try:
                        self.train_loss_array = json.load(f)
                        try:
                            self.train_loss_array.append(loss.item())
                        except AttributeError:
                            raise AttributeError
                    except json.decoder.JSONDecodeError:
                        pass
                # save the whole losses by every batch
                with open(self.train_loss_path, "a") as fp:
                    fp.write(json.dumps(self.train_loss_array))


                epoch_train_loss_avg = epoch_train_loss / len(X_train)
                with open(self.avg_train_loss_path, 'r') as f:
                    try:
                        self.avg_train_loss_array = json.load(f)
                        try:
                            self.avg_train_loss_array.append(epoch_train_loss_avg)
                        except AttributeError:
                            raise AttributeError
                    except json.decoder.JSONDecodeError:
                        pass
                # save the whole losses by every batch
                with open(self.avg_train_loss_path, "a") as fp:
                    fp.write(json.dumps(self.avg_train_loss_array))

            ### VALIDATION => validate every epoch to find and save the best model
            epoch_valid_loss = 0.0
            self.model.eval()

            with torch.no_grad():
                for X_batch, y_batch in zip(batch(X_val, self.batch_size), batch(y_val, self.batch_size)):
                    current_batch_size = len(X_batch)
                    # Forward Pass
                    batch_concat_pairs, potential_slices = self.model(X_batch)  # SIGMOID RESULTS of one batch
                    batch_outputs, _ = self.model.predict(batch_concat_pairs, potential_slices)

                    for sample_results in batch_outputs:
                        epoch_val_sample_pairs_len += len(sample_results)
                    flat_outputs = torch.cat([p for sample in batch_outputs for p in sample]).to(self.device)
                    flat_labels = torch.Tensor([l for sample in y_batch for l in sample]).to(self.device)
                    # Find the Loss
                    loss = self.criterion(flat_outputs, flat_labels)
                    # Calculate Loss
                    epoch_valid_loss += loss.item()
                    sum_val_loss += loss.item()

                    with open(self.valid_loss_path, 'r') as f:
                        try:
                            self.valid_loss_array = json.load(f)
                            try:
                                self.valid_loss_array.append(loss.item())
                            except AttributeError:
                                raise AttributeError
                        except json.decoder.JSONDecodeError:
                            pass
                    # save the whole losses by every batch
                    with open(self.valid_loss_path, "a") as fp:
                        fp.write(json.dumps(self.valid_loss_array))


                    epoch_valid_loss_avg = epoch_valid_loss / len(X_val)
                    with open(self.avg_valid_loss_path, 'r') as f:
                        try:
                            self.avg_valid_loss_array = json.load(f)
                            try:
                                self.avg_valid_loss_array.append(epoch_valid_loss_avg)
                            except AttributeError:
                                raise AttributeError
                        except json.decoder.JSONDecodeError:
                            pass
                    # save the whole losses by every batch
                    with open(self.avg_valid_loss_path, "a") as fp:
                        fp.write(json.dumps(self.avg_valid_loss_array))

            with open('../results/pretrain_{train_feature}_evaluation.txt'.format(train_feature=self.train_feature),
                      'a') as f:
                f.write(
                    'Epoch {e} \t\t Training Loss: {epoch_train_loss} \t\t Validation Loss: {epoch_valid_loss}\n\n'.format(
                        e=e + 1, epoch_train_loss=epoch_train_loss, epoch_valid_loss=epoch_valid_loss))

            early_stopping(epoch_valid_loss, self.model, self.optimizer, e)

            if early_stopping.early_stop_break_epoch == True:
                with open('../results/pretrain_{train_feature}_evaluation.txt'.format(train_feature=self.train_feature),
                          'a') as f:
                    f.write('EARLY BREAK AT EPOCH {e}'.format(e=e + 1))
                break

        # EVALUATION
        with torch.no_grad():
            success_rate = self.evaluate(self.X_test, self.y_test, 200)
        success_rate_2 = self.num_all_correct / self.num_all_anaphor

        with open('../results/pretrain_{train_feature}_evaluation.txt'.format(train_feature=self.train_feature),
                  'a') as f:
            f.write('\n\n Success rate: {success_rate}'.format(success_rate=success_rate))
            f.write('\n\n Success rate2: {success_rate}'.format(success_rate=success_rate_2))


        with open(self.eva_path, "w") as fp:
            fp.write(json.dumps(str(success_rate) + '\n' + 'success rate 2: ' + str(success_rate_2)))

    # EVALUATE the model
    def evaluate(self, X_test, y_test, num_evaSamples_to_save):
        '''
        Evaluate the model on the test set
        @return: Success rate = successfully resolved anaphors/number of all anaphors
        '''
        print('Evaluating...')
        ana_num_id = 0
        if ana_num_id < num_evaSamples_to_save:
            eva_file = open(
                '../evaluation/pretrain_{train_feature}_evaluation.txt'.format(train_feature=self.train_feature), 'a')
        all_ana_len = 0
        self.model.eval()
        correct = 0
        num_anaphors = 0

        # FOR EVERY ANAPHOR
        for X_batch, y_batch in zip(batch(X_test, self.batch_size), batch(y_test, self.batch_size)):
            current_batch_size = len(X_batch)
            all_ana_len += current_batch_size

            batch_concat_pairs, potential_slices = self.model(X_batch)  # SIGMOID RESULTS of one batch
            batch_label_outputs, _ = self.model.predict(batch_concat_pairs, potential_slices)

            ana_strs = [ana["tokens"] for ana in X_batch]
            ana_contexts = [ana['context'] for ana in X_batch]
            golds_str = [ana["golds_str"] for ana in X_batch]
            all_ps_strs = [[p["tokens"] for p in ana["potential_antecedents"]] for ana in X_batch]
            # EVERY SAMPLE
            i = 0
            for targets, ys in zip(batch_label_outputs, y_batch):
                num_anaphors += 1
                selected_index = targets.index(max(targets))
                if ana_num_id < num_evaSamples_to_save:
                    eva_file.write('anaphor: ' + str(ana_strs[i]) + '\n')
                    eva_file.write('candidates: ' + str(all_ps_strs[i]) + '\n')
                    eva_file.write('gold: ' + str(golds_str[i]) + '\n')
                    eva_file.write('selected cand: ' + str(all_ps_strs[i][selected_index]) + '\n')
                    eva_file.write('context: ' + str(ana_contexts[i]) + '\n')

                # if one of the candidates is gold
                if 1 in ys:  # because its possible that none of the potential candicates is right(e.g. the gold antecedent is not in the 3 sents before)
                    gold_and_coref_idxs = [i for i, val in enumerate(ys) if val == 1]
                    if selected_index in gold_and_coref_idxs:
                        correct += 1
                        if ana_num_id < num_evaSamples_to_save:
                            eva_file.write('right\n')
                    else:
                        if ana_num_id < num_evaSamples_to_save:
                            eva_file.write('wrong\n')
                else:
                    if ana_num_id < num_evaSamples_to_save:
                        eva_file.write('wrong\n')
                if ana_num_id < num_evaSamples_to_save:
                    eva_file.write('\n')
                i += 1
            ana_num_id += 1
        self.num_all_correct += correct
        self.num_all_anaphor += num_anaphors
        success_rate = correct / num_anaphors

        self.success_rate = success_rate
        with open('../results/pretrain_{train_feature}_evaluation.txt'.format(train_feature=self.train_feature), 'a') as f:
            f.write('Anaphor ammounts: ' + str(num_anaphors))
            f.write('Success rate: ' + str(success_rate))
        if ana_num_id < num_evaSamples_to_save:
            eva_file.close()

        return success_rate

if __name__ == '__main__':
    print('Loading datasets...')

    train_val_test = load_corpus_list()

    reduced_size_train = round(len(train_val_test[0][0])/5)
    reduced_size_test = round(len(train_val_test[1][0])/5)
    reduced_size_val = round(len(train_val_test[2][0])/5)

    X_train = train_val_test[0][0][:reduced_size_train]
    X_val = train_val_test[1][0][:reduced_size_test]
    X_test = train_val_test[2][0][:reduced_size_val]

    y_train = train_val_test[3][:-1][:reduced_size_train]
    y_val = train_val_test[4][:-1][:reduced_size_test]
    y_test = train_val_test[5][:-1][:reduced_size_val]

    early_stopping_patience = 10
    num_epochs = 100  # klowersa = 20
    learning_rate = 1e-6
    batch_size = 8  # the number of training examples utilized in one iteration = length of training samples
    elmo_emb_size = 256
    glove_emb_size = 300
    embedding_size = elmo_emb_size + glove_emb_size
    hidden_size = embedding_size  # Klowersa: hidden size of the BiLSTM = word embedding dimensionality

    train_feature = '5DATA_4layer_5DATA_2dropout_NoLN_NoFea'

    model = Model(batch_size=batch_size,
                  num_layers=1,
                  distance_feature=False,
                  grammar_role_feature=False,
                  definiteness_feature=False,
                  match_feature=False,
                  synonym_feature=False,
                  hypernym_feature=False)

    trainer = Trainer(train_feature,
                      model,
                      early_stopping_patience,
                      learning_rate,
                      num_epochs,
                      batch_size,
                      X_train,
                      X_val,
                      X_test,
                      y_train,
                      y_val,
                      y_test)

    trainer.train()