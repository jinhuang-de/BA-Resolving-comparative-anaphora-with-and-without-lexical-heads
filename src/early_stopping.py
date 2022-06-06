import torch
import numpy as np

class EarlyStopping:
    # adapted from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

    """Early stops the training if validation loss doesn't improve after a given patience.
    => Stop training as soon as the error on the validation set is higher than it was the last time it was checked."""
    def __init__(self, train_feature, corpus_name=None, patience=7, verbose=False, pretrain=False, delta=0, save_model=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.global_best_score = None
        self.fold_best_score = None
        self.pretrain = pretrain
        self.save_model = save_model

        self.val_loss_min = np.Inf
        self.delta = delta

        self.train_feature = train_feature
        self.corpus_file_name = corpus_name

        self.early_stop_break_epoch = False

    def __call__(self, val_loss, model, optimizer, epoch_id):
        score = val_loss
        # initialize global fold lowest validation loss
        if self.global_best_score is None:
            self.global_best_score = score
            if self.save_model == True:
                self.save_checkpoint(val_loss, model, optimizer, epoch_id)

        ### if its pre-training
        if self.pretrain == True:
            if score > self.global_best_score + self.delta:
                self.counter += 1
                # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter > self.patience:
                    self.early_stop_break_epoch = True
            # if the validation loss is lower than it was
            # save the model as the best model
            else:
                self.global_best_score = score
                if self.save_model == True:
                    self.save_checkpoint(val_loss, model, optimizer, epoch_id)
                self.counter = 0

        ### if its BiLSTM training
        # or
        # Transfer learning training
        if self.pretrain == False:
            # initialize local fold lowest validation loss
            if self.fold_best_score is None:
                self.fold_best_score = score
            # safe the best model
            if score < self.global_best_score:
                self.global_best_score = score
                if self.save_model == True:
                    self.save_checkpoint(val_loss, model, optimizer, epoch_id)

            # if the validation loss is higher than the lowest validation loss in one fold, stop the fold
            if score > self.fold_best_score + self.delta:
                self.counter += 1
                # if more than patience times
                # => EARLY STOP!!!(stop the training of the fold
                #  set fold_best_score to none to be initialized in next fold
                if self.counter >= self.patience:
                    self.early_stop_break_epoch = True
                    self.fold_best_score = None
                    self.counter = 0
            else:
                self.fold_best_score = score
                self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch_id):
        '''Saves model when validation loss decrease.'''
        # torch.save(model.state_dict(), self.path)
        # Save the model
        if self.corpus_file_name is not None:
            torch.save(model, '../trained_model/best_model_{corpus_name}_{train_feature}.pth'.format(
                train_feature=self.train_feature, corpus_name=self.corpus_file_name))

            # Save State Dict
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch_id,
                        'optimizer_state_dict': optimizer.state_dict()},
                       '../trained_model/best_state_dict_{corpus_name}_{train_feature}.pth'.format(
                           train_feature=self.train_feature, corpus_name=self.corpus_file_name))

        else:
            torch.save(model, '../trained_model/best_model_ontonotes_pretraining_2_{train_feature}.pth'.format(train_feature=self.train_feature))

            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch_id,
                        'optimizer_state_dict': optimizer.state_dict()},
                        '../trained_model/best_state_dict_ontonotes_pretraining_2_{train_feature}.pth'.format(train_feature=self.train_feature))

        self.val_loss_min = val_loss