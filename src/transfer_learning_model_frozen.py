from torch import nn
from loader import *

class transfer_learning_model(nn.Module):
    def __init__(self):
        super(transfer_learning_model, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pretrained_model = torch.load('../trained_model/old_pretrained_model.pth', map_location=torch.device(self.device))
        # print(self.pretrained_model.ffnn)
        self.pretrained_model.pretraining=False
        # change corpus vocabulary for glove embeddings
        self.pretrained_model.whole_corpus = Corpus().corpus_total
        self.pretrained_model.dropout.p=0.3

        # FREEZE ALL LAYERS
        self.pretrained_model.train(False)
        idx= 0
        for c, p in zip(self.pretrained_model.named_children(), self.pretrained_model.parameters()):
            if c[0] == 'ffnn':
                for f_c, f_p in zip(c[1].named_children(), c[1].parameters()):
                    f_p.requires_grad = False
                    print(f_c, f_p.requires_grad)
                    print(f_c[1], type(f_c[1]))
                    if isinstance(f_c[1], torch.nn.modules.dropout.Dropout):
                        f_c[1].p = 0
            else:
                p.requires_grad = False
                # print(c, p.requires_grad)
            idx += 1
        # ---------FFNN MODIFICATION---------
        # replace the last 2 layers in the FFNN
        # this two layers are not frozen
        self.pretrained_model.ffnn[8] = nn.ReLU()
        self.pretrained_model.ffnn[9] = nn.Linear(64, 1)

        # replace the prediction which contains the sigmoid
        self.pretrained_model.predict = self.predict
        # ------------------------------------------------
        ###########print#############
        print('\n\nafter delete....\n')
        for c, p in zip(self.pretrained_model.named_children(), self.pretrained_model.parameters()):
            if c[0] == 'ffnn':
                for f_c, f_p in zip(c[1].named_children(), c[1].parameters()):
                    print(f_c, f_c[1], f_p.requires_grad)
            idx += 1
        ########################

    def predict(self, batch_concat_pairs, potential_slices):
        # FFNN PREDICTION
        results_sigmoid = []
        results_labels = []
        for pairs_pro_sample, p_slices_pro_sample in zip(batch_concat_pairs, potential_slices):
            sigmoid_results_pro_sample = []
            labels_reults_pro_sample = []
            for pair in pairs_pro_sample:
                ffnn_out = self.pretrained_model.ffnn(pair)
                score = self.sigmoid(ffnn_out)
                sigmoid_results_pro_sample.append(score)  # different len because of different amount of potentials: list of potentials([512])
                labels_reults_pro_sample.append(assign_label(score))
            results_sigmoid.append(sigmoid_results_pro_sample)
            results_labels.append(labels_reults_pro_sample)
        return results_sigmoid, results_labels  # len = batch size :  list of lists of potentials scores(tensors with one float e.g. tensor([0.27]))

    def forward(self, x):#, current_batch_size
        x = self.pretrained_model.forward(x)# current_batch_size
        return x