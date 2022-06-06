import pandas as pd
import spacy
import inflect
from spacy.tokens import Doc
import ast
import re, unicodedata
from nltk.corpus import stopwords

class preoprocessing:
    def __init__(self):
        self.corpus_path = '/Users/huangjin/Desktop/Studium/Sebtes Semester/BA/corpus/Dataset_comparative_anaphora_resolution/'
        self.df = pd.read_csv(self.corpus_path + 'preprocessed/annotation_retrieval.csv', sep='\t', index_col=[0])  # get rid of the 'Unnamed' caused by index
        self.df_with = pd.read_csv(self.corpus_path + 'preprocessed/anno_with_lexical_info.csv', sep='\t', index_col=[0])
        self.df_without = pd.read_csv(self.corpus_path + 'preprocessed/anno_without_lexical_info.csv', sep='\t', index_col=[0])
        self.nlp = spacy.load("en_core_web_sm")

    def custom_spacy_tokenizer(self, text):
        return Doc(self.nlp.vocab, words=text)

    # Text normalization includes many steps.
    def nomalization(self, tokenized_sentences):
        new_tokenized_sentences = []
        p = inflect.engine()
        self.nlp.tokenizer = self.custom_spacy_tokenizer
        doc = self.nlp(tokenized_sentences)

        for word in doc:
            print(word.lemma_)
            # Remove non-ASCII characters from list of tokenized words
            new_word = word.lemma_
            new_word = unicodedata.normalize('NFKD', new_word).encode('ascii', 'ignore').decode('utf-8', 'ignore')

            # Lowercase
            new_word = new_word.lower()

            # Replace all interger occurrences in list of tokenized words with textual representation
            if new_word.isdigit():
                new_word = p.number_to_words(new_word)

            # Remove stop words
            if new_word not in stopwords.words('english'):
                # Remove punctuation
                new_word = re.sub(r'[^\w\s]', '', new_word)
                # TODO: INDEX PROBLEM
                if new_word != '':
                    new_tokenized_sentences.append(new_word)

        print(new_tokenized_sentences)

        return new_tokenized_sentences


    def main(self):

        for index, row in self.df.iterrows():
            context_dict = ast.literal_eval(row['context'])
            potential_count_pro_anaphor = context_dict

        normalize.nomalization(sent)

if __name__ == '__main__':
    sent = ['Tom','\'s', 'odered', 'ice', 'cream!']
    normalize = preoprocessing()
    normalize.main()