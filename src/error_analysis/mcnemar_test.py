import re
import numpy as np
# from mlxtend.evaluate import mcnemar_table
from statsmodels.stats.contingency_tables import mcnemar

def my_mcnemar(path_1, path_2):
    def results_list(path):
        with open(path) as f:
            eva = f.read()

        result_list = re.findall(r'wrong\n|right\n', eva)
        result = [0 if r == 'wrong\n' else 1 for r in result_list  ]
        return result

    def contingency_table(list_1, list_2):
        _1_right_2_right = 0
        _1_right_2_wrong = 0
        _1_wrong_2_right = 0
        _1_wrong_2_wrong = 0

        for y_1, y_2 in zip(list_1, list_2):
            if y_1 == 1:
                # 1_right_2_right
                if y_2 == 1:
                    _1_right_2_right += 1
                # 1_right_2_wrong
                else:
                    _1_right_2_wrong += 1
            else:
                # 1_wrong_2_right
                if y_2 == 1:
                    _1_wrong_2_right += 1
                # 1_wrong_2_wrong
                else:
                    _1_wrong_2_wrong += 1

        table = [[_1_right_2_right, _1_wrong_2_right],
                 [_1_right_2_wrong, _1_wrong_2_wrong]]

        table_to_print = [[_1_right_2_right, _1_wrong_2_right, _1_right_2_right + _1_wrong_2_right], [_1_right_2_wrong, _1_wrong_2_wrong, _1_right_2_wrong + _1_wrong_2_wrong]
                          ,[_1_right_2_right + _1_right_2_wrong, _1_wrong_2_right + _1_wrong_2_wrong, _1_right_2_right + _1_right_2_wrong+ _1_wrong_2_right + _1_wrong_2_wrong]]
        return table, table_to_print

    result_1 = results_list(path_1)
    result_2 = results_list(path_2)

    table, table_to_print = contingency_table(result_1, result_2)

    for l in table_to_print:
        print(l)
    result = mcnemar(table)
    print(result)

def print_mcnemar(path_1, path_2, name_1, name_2):
    print(f'{name_1} vs {name_2}'.format(name_1, name_2))
    my_mcnemar(path_1, path_2)
    print('\n')

if '__main__'==__name__:
    print('\n======= WithoutLex - RECENCY vs ALL MODELS  =======')
    # recency vs elmo
    path_1 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/recency_baseline_withoutLex_evaluation.txt'

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withoutlex/anaphor_without_newest_only_elmo_noPool_sum_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithoutLex - RECENCY", "ELMo")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withoutlex/anaphor_without_newest_elmo_6fea_noPool_sum_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithoutLex - RECENCY", "ELMo + 6fea")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withoutlex/anaphor_without_bert_elmo_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithoutLex - RECENCY", "ELMo + spanBERT")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_glove_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithoutLex - RECENCY", "BiLSTM GloVe")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_glove_6fea_noPooling_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithoutLex - RECENCY", "BiLSTM GloVe + 6fea")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_elmo_noPooling_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithoutLex - RECENCY", "BiLSTM ELMo")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_elmo_6fea_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithoutLex - RECENCY", "BiLSTM ELMo + 6fea")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_elmo_glove_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithoutLex - RECENCY", "BiLSTM ELMo + GloVe")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_elmo_glove_6fea_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithoutLex - RECENCY", "BiLSTM ELMo + GloVe + 6fea")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/anaphor_without_bilstm_bert_elmo_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithoutLex - RECENCY", "BiLSTM ELMo + spanBERT")

    # ====================================================================================
    print('=======WithLex - SPANBERT VS all WithLex =======')
    # elmo with vs without feature
    path_1 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bert_cosine_baseline_WithLex.txt'
    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withlex/anaphor_with_newest_only_elmo_noPool_sum_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithLex - spanBERT", "ELMo")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withlex/anaphor_with_newest_elmo_6fea_noPool_sum_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithLex - spanBERT", "ELMo + 6fea")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withlex/anaphor_with_bert_elmo_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithLex - spanBERT", "ELMo + spanBERT")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_glove_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithLex - spanBERT", "BiLSTM GloVe")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_glove_6fea_noPooling_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithLex - spanBERT", "BiLSTM GloVe + 6fea")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_elmo_noPooling_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithLex - spanBERT", "BiLSTM ELMo")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_elmo_6fea_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithLex - spanBERT", "BiLSTM ELMo + 6fea")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_elmo_glove_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithLex - spanBERT", "BiLSTM ELMo + GloVe")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_elmo_glove_6fea_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithLex - spanBERT", "BiLSTM ELMo + GloVe + 6fea")

    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/anaphor_with_bilstm_bert_elmo_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithLex - spanBERT", "BiLSTM ELMo + spanBERT")


    print('======= WithLex - WITHOUT vs WITH FEATURE - ALL MODELS=======')
    # elmo with vs without feature
    path_1 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withlex/anaphor_with_newest_only_elmo_noPool_sum_evaluation.txt'
    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withlex/anaphor_with_newest_elmo_6fea_noPool_sum_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithLex - ELMo", "ELMo + 6fea")


    path_1 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_glove_evaluation.txt'
    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_glove_6fea_noPooling_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithLex - BiLSTM GloVe", "GloVe + 6fea")

    path_1 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_elmo_noPooling_evaluation.txt'
    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_elmo_6fea_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithLex - BiLSTM ELMo", "ELMo + 6fea")

    path_1 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_elmo_glove_evaluation.txt'
    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_with_bilstm_elmo_glove_6fea_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithLex - BiLSTM ELMo + GloVe", "ELMo + GloVe + 6fea")
    #====================================================================================

    print('======= WithoutLex - WITHOUT vs WITH FEATURE - ALL MODELS =======')
    # elmo with vs without feature
    path_1 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withoutlex/anaphor_without_newest_only_elmo_noPool_sum_evaluation.txt'
    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withoutlex/anaphor_without_newest_elmo_6fea_noPool_sum_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithoutLex - ELMo", "ELMo + 6fea")

    # bilstm glove with vs without feature
    path_1 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_glove_evaluation.txt'
    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_glove_6fea_noPooling_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithoutLex - BiLSTM GloVe", "GloVe + 6fea")

    path_1 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_elmo_noPooling_evaluation.txt'
    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_elmo_6fea_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithoutLex - BiLSTM ELMo", "ELMo + 6fea")

    path_1 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_elmo_glove_evaluation.txt'
    path_2 = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/bilstm/anaphor_without_bilstm_elmo_glove_6fea_evaluation.txt'
    print_mcnemar(path_1, path_2, "WithoutLex - BiLSTM ELMo + GloVe", "ELMo + GloVe + 6fea")


    print('======= WithoutLex - transfer vs elmo ==========')

    path_elmo = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/elmo/withoutlex/anaphor_without_newest_only_elmo_noPool_sum_evaluation.txt'

    path_finetune = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/anaphor_without_NEW_FINAL_transfer_learning_finetune_evaluation.txt'
    print_mcnemar(path_elmo, path_finetune, "WithoutLex - ELMO", "Transfer(Finetune)")

    path_frozen = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/BA_project/evaluation/anaphor_without_NEW_FINAL_transfer_learning_frozen_evaluation.txt'
    print_mcnemar(path_elmo, path_frozen, "WithoutLex - ELMO", "Transfer(Freeze)")

