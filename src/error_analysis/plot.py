import matplotlib.pyplot as plt
import json
import os
import numpy as np
import ast
def load_array(path):
    with open(path, 'r') as f:
        array = json.load(f)
    return array

'''
if __name__ == '__main__':
    path = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/src/arrays/2_embeddings_6_features/with/anaphor_with_2_embeddings_6_features_train_loss.txt'

    corpura = ['with', 'without', 'total'] # , 'total'
    features = ['elmo_6fea', 'elmo_6fea_layernorm_adamW'] #'elmo_6_fea pointwise'
    values = ['train_loss', 'valid_loss', 'eva']
    # i = 0
    for feature in features:
        for corpus in corpura:
            fh = plt.figure()
            for value in values:
                # print(corpus, feature, value)
                if corpus == 'total':
                    path = "/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/src/arrays/total_{feature}_{value}.txt".format(corpus=corpus, feature=feature,value=value)
                else:
                    path = "/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/src/arrays/anaphor_{corpus}_{feature}_{value}.txt".format(corpus=corpus, feature=feature,value=value)
                array = np.array(load_array(path))
                plt.plot(array, label=value)
                label_name = value.split('_')[0]

            plt.legend()
            plt.ylabel(corpus + '_' + feature + '_' + value)
            plt.savefig('../plot/' + corpus + '_' + feature + '_' + value)
            plt.close(fh)

    # Pretrain
    p = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/src/arrays/pretrain/'

    # TODO: CHANGE TO THIS(+EVA)
    # pathes = ['pretrain_2_embeddings_6_features_train_loss.txt', 'pretrain_2_embeddings_6_features_valid_loss.txt', 'pretrain_2_embeddings_6_features_eva.txt']
    pathes = ['pretrain_2_embeddings_6_features_train_loss.txt', 'pretrain_2_embeddings_6_features_valid_loss.txt']

    i = 0
    for path in pathes:
        if i == 0:
            label = 'train_loss'
        elif i == 1:
            label = 'valid_loss'
        else:
            label = 'eva'
        # array = load_array(p+path)

        with open(p+path, 'r') as f:
            array = ast.literal_eval(f.read())

        plt.plot(array, label=label)
        i += 1
        plt.legend()
    plt.ylabel('pre-train with 2 emb & 6 fea')
    plt.savefig('../plot/pretrain_6_fea')
'''


p = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/src/arrays/'
# TRAINING LOSS
fh = plt.figure()
wi = np.array(load_array(p + 'anaphor_with_elmo_6fea_layernorm_adamW_train_loss.txt'))

without = np.array(load_array(p + 'anaphor_without_elmo_6fea_layernorm_adamW_train_loss.txt'))
plt.plot(wi, label='without')


plt.legend()
plt.xlabel('epoches(5 fold validation)')
plt.ylabel('losses')
plt.savefig('../plot/' + 'elmo_6fea_layernorm_adamW_train')
plt.close(fh)

# Validation LOSS
fh = plt.figure()
wi = np.array(load_array(p + 'anaphor_with_elmo_6fea_layernorm_adamW_valid_loss.txt'))
plt.plot(wi, label='with')

without = np.array(load_array(p + 'anaphor_without_elmo_6fea_layernorm_adamW_valid_loss.txt'))
plt.plot(wi, label='without')

plt.legend()
plt.xlabel('epoches(5 fold validation)')
plt.ylabel('losses')
plt.savefig('../plot/' + 'elmo_6fea_layernorm_adamW_valid')
plt.close(fh)