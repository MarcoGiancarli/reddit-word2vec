__author__ = 'Marco Giancarli -- m.a.giancarli@gmail.com'


from nltk import tokenize
from gensim.models import word2vec
import os


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

    with open('res/text', 'w') as training_file:
        for comment_file_name in comment_files:
            with open(comment_file_name, 'r') as comment_file:
                for line in comment_file:
                    sentences = comment_to_sentences(line)
                    training_file.writelines(sentences)
                    ### Just for testing:
                    # print '-' * 40 + '\n', line
                    # print '-' * 10 + '\n', '\n'.join(sentences)

        for lmb_file_name in lmb_files:
            with open(lmb_file_name, 'r') as lmb_file:
                training_file.writelines(lmb_file)


def comment_to_sentences(comment_line):
    # format the comment text
    comment_text = ','.join(comment_line.split(',')[5:])
    comment_text = comment_text.strip()
    if len(comment_text) < 1:
        return []
    if comment_text[0] == '"' and comment_text[-1] == '"':
        comment_text = comment_text[1:-1]
    comment_text = comment_text.replace('""', '"')  # formatting from html

    sentences = tokenize.sent_tokenize(comment_text)
    sentences = [sentence + '\n' for sentence in sentences]

    return sentences


def train():
    with open('res/text', 'r') as train_file:
        model = word2vec.Word2Vec(
            sentences=train_file,
            min_count=20,
            workers=4,
        )
    model.save('text-no-phrases.bin')


if __name__ == '__main__':
    main()