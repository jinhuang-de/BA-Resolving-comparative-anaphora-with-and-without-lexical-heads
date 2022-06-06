from util import *
from transfer_learning_model_frozen import *
from new_train import *

if __name__ == '__main__':
    corpus_names = ['anaphor_without']
    i = 1

    # Train 3 corpus
    for corpus_name in corpus_names:
        train_feature = 'OLD_NEW_FINAL_transfer_learning_frozen'  # 2_embeddings_6_features   |   2_embeddings   |   only_elmo   |   2_emb_only_distance_fea

        early_stopping_patience = 8
        num_epochs = 80  # klowersa = 20
        batch_size = 16
        learning_rate = 1e-6
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
                         # if_bert=True,

                         glove=False,
                         if_elmo=False,
                         save_model=False,
                         transfer_learning=True)
        trainer.model = transfer_learning_model()


        trainer.train()
        i += 1
