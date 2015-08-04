__author__ = 'Marco Giancarli -- m.a.giancarli@gmail.com'


import os
import re
from nltk.tokenize import sent_tokenize

from gensim.models import phrases
from gensim.models import word2vec


INPUT_FILE = 'res/sentences.txt'
OUTPUT_FILE = 'res/reddit-w2v.bin'

comments_dir = 'res/comments/'
lmb_heldout_dir = 'res/language-modeling-benchmark/heldout/'
lmb_training_dir = 'res/language-modeling-benchmark/training/'


def main():
    # make_training_file()
    train()


def make_training_file():
    comment_files = []
    lmb_files = []

    # we need to grab and format the comment text before adding it
    comment_files += [
        comments_dir + filename
        for filename
        in os.listdir(comments_dir)
    ]
    # these have one sentence per line already, so just add each line
    lmb_files += [
        lmb_heldout_dir + filename
        for filename
        in os.listdir(lmb_heldout_dir)
    ]
    lmb_files += [
        lmb_training_dir + filename
        for filename
        in os.listdir(lmb_training_dir)
    ]

    with open(INPUT_FILE, 'w') as training_file:
        for comment_file_name in comment_files:
            with open(comment_file_name, 'r') as comment_file:
                for line in comment_file:
                    # minimize callstack by calling .lower() now
                    sentences = comment_to_sentences(line.lower())
                    formatted_sentences = [
                        re.sub(' +',' ', ''.join([
                            c for c in sentence
                            if str.isalnum(c) or c == ' ']
                        ) + '\n')
                        for sentence in sentences
                    ]
                    training_file.writelines(formatted_sentences)
                    ### Just for testing:
                    # print '-' * 40 + '\n', line
                    # print '-' * 10 + '\n', '\n'.join(formatted_sentences)

        for lmb_file_name in lmb_files:
            with open(lmb_file_name, 'r') as lmb_file:
                for sentence in lmb_file:
                    formatted_sentence = re.sub(' +',' ', ''.join(
                        c for c in sentence.lower()
                        if str.isalnum(c) or c == ' ')
                    )
                    training_file.writelines(formatted_sentence)
                    ### Just for testing:
                    # print '-' * 40
                    # print sentence
                    # print formatted_sentence


def comment_to_sentences(comment_line):
    # format the comment text
    comment_text = ','.join(comment_line.split(',')[5:])
    comment_text = comment_text.strip()
    if len(comment_text) < 1:
        return []
    if comment_text[0] == '"' and comment_text[-1] == '"':
        comment_text = comment_text[1:-1]
    comment_text = comment_text.replace('""', '"')  # formatting from html

    sentences = sent_tokenize(comment_text)
    sentences = [sentence + '\n' for sentence in sentences]

    return sentences


def train():
    choo_choo_train = word2vec.LineSentence(INPUT_FILE)
    bigram = phrases.Phrases(
        sentences=choo_choo_train,
        min_count=50,
        threshold=10.0,
    )
    trigram = phrases.Phrases(
        sentences=bigram[choo_choo_train],
        min_count=50,
        threshold=10.0,
    )
    model = word2vec.Word2Vec(
        sentences=trigram[choo_choo_train],
        min_count=100,
        size=100,
        workers=4,
    )
    model.init_sims(replace=True)
    model.save(OUTPUT_FILE)


if __name__ == '__main__':
    main()