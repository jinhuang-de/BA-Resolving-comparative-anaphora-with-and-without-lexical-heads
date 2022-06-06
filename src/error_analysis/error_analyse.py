import re
import numpy as np
# from mlxtend.evaluate import mcnemar_table
from operator import itemgetter

def find_error_analyse_sample(path_1, path_2, model_names, *pathes):
    def results_list(path):
        with open(path) as f:
            eva = f.read()
        result_list = re.findall(r'wrong\n|right\n', eva)
        result = [0 if r == 'wrong\n' else 1 for r in result_list  ]
        return result

    def find_sample_idxs(result_lists, list_1, list_2):
        all_right_idxs = []
        all_wrong_idxs = []

        a_right_b_wrong = []
        b_right_a_wrong = []

        idx = 0
        # for all models
        for one_samp_all_labs in zip(*result_lists):
            if 0 not in one_samp_all_labs:
                all_right_idxs.append(idx)
            elif 1 not in one_samp_all_labs:
                all_wrong_idxs.append(idx)
            idx += 1

        # for the two chosen models
        idx = 0
        for pair in zip(list_1, list_2):
            if pair[0] == 0 and pair[1] == 1:
                b_right_a_wrong.append(idx)
            if pair[0] == 1 and pair[1] == 0:
                a_right_b_wrong.append(idx)
            idx += 1

        print('all_right: ', all_right_idxs)
        print('all_wrong: ', all_wrong_idxs)
        print('a_right_b_wrong: ', a_right_b_wrong)
        print('b_right_a_wrong: ', b_right_a_wrong)

        return all_right_idxs, all_wrong_idxs, a_right_b_wrong, b_right_a_wrong

    def read_txt(path):
        with open(path) as f:
            return f.read()

    def find_all_model_samples(idxs, *pathes):

        evas = [re.split(r'wrong\n|right\n', read_txt(path)) for path in pathes]
        model_samples = [[s for i, s in enumerate(samples) if i in idxs] for samples in evas]

        for name, samples in zip(model_names, model_samples):
            print('\n=================' + name + "=================\n")
            # for s in samples:
            #     print(s)
            print(samples[2])
        # print('\n==============================\n==============================\n')


        # return samples_1, samples_2

    def find_sample(path_1, path_2, idxs): # , right=True
        eva_1 = read_txt(path_1)
        eva_2 = read_txt(path_2)
        # if right == True:
        samples_1 = re.split(r'wrong\n|right\n', eva_1)
        samples_2 = re.split(r'wrong\n|right\n', eva_2)
        samples_1 = [s for i, s in enumerate(samples_1) if i in idxs]
        samples_2 = [s for i, s in enumerate(samples_2) if i in idxs]

        for s in samples_1:
            print(s)

        print('\n==============================\n==============================\n')

        for s in samples_2:
            print(s)

        return samples_1, samples_2


    result_lists = [results_list(path) for path in pathes]

    # print('result_lists: ', result_lists)
    list_1 = results_list(path_1)
    # print(path_2)
    list_2 = results_list(path_2)
    all_right_idxs, all_wrong_idxs, a_right_b_wrong, b_right_a_wrong = find_sample_idxs(result_lists, list_1, list_2)


    # find_sample(path_1, path_2, a_right_b_wrong)

    find_all_model_samples(a_right_b_wrong, *pathes)


if '__main__'==__name__:
    # print('\n======= WithoutLex - RECENCY VS ELMO MODEL ON: =======')

    recency_without_path = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/recency_baseline_withoutLex_evaluation.txt'
    recency_with_path = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/recency_baseline_withLex_evaluation.txt'

    with_bert_baseline = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/with_span_BERT.txt'
    without_bert_baseline = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/without_span_BERT.txt'

    with_elmo = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withlex/anaphor_with_newest_only_elmo_noPool_sum_evaluation.txt'
    with_elmo_6fea = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withlex/anaphor_with_newest_elmo_6fea_noPool_sum_evaluation.txt'
    with_elmo_bert = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withlex/anaphor_with_bert_elmo_evaluation.txt'

    without_elmo = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withoutlex/anaphor_without_newest_only_elmo_noPool_sum_evaluation.txt'
    without_elmo_6fea = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withoutlex/anaphor_without_newest_elmo_6fea_noPool_sum_evaluation.txt'
    without_elmo_bert = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withoutlex/anaphor_without_bert_elmo_evaluation.txt'


    with_bilstm_glove = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_glove_evaluation.txt'
    with_bilstm_glove_6fea = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_glove_6fea_noPooling_evaluation.txt'
    with_bilstm_elmo = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_elmo_noPooling_evaluation.txt'
    with_bilstm_elmo_6fea = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_elmo_6fea_evaluation.txt'
    with_bilstm_elmo_glove = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_elmo_glove_evaluation.txt'
    with_bilstm_elmo_glove_6fea = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_elmo_glove_6fea_evaluation.txt'
    with_bilstm_elmo_bert = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/anaphor_with_bilstm_bert_elmo_evaluation.txt'


    without_bilstm_glove = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_glove_evaluation.txt'
    without_bilstm_glove_6fea = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_glove_6fea_noPooling_evaluation.txt'
    without_bilstm_elmo = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_elmo_noPooling_evaluation.txt'
    without_bilstm_elmo_6fea = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_elmo_6fea_evaluation.txt'
    without_bilstm_elmo_glove = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_elmo_glove_evaluation.txt'
    without_bilstm_elmo_glove_6fea = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_elmo_glove_6fea_evaluation.txt'
    without_bilstm_elmo_bert = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/anaphor_without_bilstm_bert_elmo_evaluation.txt'


    all_with_model_names = [# "recency_with_path",
                      # "with_bert_baseline",
                      "with_elmo",
                      "with_elmo_6fea",
                      "with_elmo_bert",
                      "with_bilstm_glove",
                      "with_bilstm_glove_6fea",
                      "with_bilstm_elmo",
                      "with_bilstm_elmo_6fea",
                      "with_bilstm_elmo_glove",
                      "with_bilstm_elmo_glove_6fea",
                    "with_bilstm_elmo_bert"]

    all_with_elmo_names = ["with_elmo",
                      "with_elmo_6fea",
                      "with_elmo_bert"]

    all_with_bilstm_names = ["with_bilstm_glove",
                      "with_bilstm_glove_6fea",
                      "with_bilstm_elmo",
                      "with_bilstm_elmo_6fea",
                      "with_bilstm_elmo_glove",
                      "with_bilstm_elmo_glove_6fea",
                             "with_bilstm_elmo_bert"]

    all_without_model_names = [# "recency_without_path",
                           # "with_bert_baseline",
                            "without_elmo",
                            "without_elmo_6fea",
                            "without_elmo_bert",
                            "without_bilstm_glove",
                            "without_bilstm_glove_6fea",  # 38?
                            "without_bilstm_elmo",
                            "without_bilstm_elmo_6fea",  # 75
                            "without_bilstm_elmo_glove_6fea",
                            "without_bilstm_elmo_glove",
                "without_bilstm_elmo_bert"]

    all_without_elmo_names = ["without_elmo",
                       "without_elmo_6fea",
                       "without_elmo_bert"]

    all_without_bilstm_names = [# "without_bilstm_glove",
                         # "without_bilstm_glove_6fea",
                         "without_bilstm_elmo",
                         "without_bilstm_elmo_6fea",
                         "without_bilstm_elmo_glove",
                         "without_bilstm_elmo_glove_6fea",
                        " without_bilstm_elmo_bert"]

    with_all_pathes = [
              # recency_with_path,
              # with_bert_baseline,
              with_elmo,
              with_elmo_6fea,
              with_elmo_bert,
              with_bilstm_glove,
              with_bilstm_glove_6fea,
              with_bilstm_elmo,
              with_bilstm_elmo_6fea,
              with_bilstm_elmo_glove,
              with_bilstm_elmo_glove_6fea,
              with_bilstm_elmo_bert]

    all_with_elmo = [with_elmo,
              with_elmo_6fea,
              with_elmo_bert]

    all_with_bilstm = [with_bilstm_glove,
              with_bilstm_glove_6fea,
              with_bilstm_elmo,
              with_bilstm_elmo_6fea,
              with_bilstm_elmo_glove,
              with_bilstm_elmo_glove_6fea,
                with_bilstm_elmo_bert]

    without_all_pathes = [# recency_without_path,
                          # without_bert_baseline,
                        without_elmo,
                        without_elmo_6fea,
                        without_elmo_bert,
                        without_bilstm_glove,
                        without_bilstm_glove_6fea,  # 38?
                        without_bilstm_elmo,
                        without_bilstm_elmo_6fea,  # 75
                        without_bilstm_elmo_glove_6fea,
                        without_bilstm_elmo_glove,
                        without_bilstm_elmo_bert]

    all_without_elmo = [without_elmo,
                 without_elmo_6fea,
                 without_elmo_bert]

    all_without_bilstm = [without_bilstm_glove,
                   without_bilstm_glove_6fea,
                   without_bilstm_elmo,
                   without_bilstm_elmo_6fea,
                   without_bilstm_elmo_glove,
                   without_bilstm_elmo_glove_6fea,
                   without_bilstm_elmo_bert]

    compair_2_pathes = [with_bert_baseline, with_bilstm_elmo_bert]
    # print(with_elmo)
    find_error_analyse_sample(without_bert_baseline, without_elmo, all_without_model_names, *compair_2_pathes)