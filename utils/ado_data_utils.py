# -*- coding:utf-8 -*-
"""
@contact: adonis_wu@outlook.com
@file: ado_data_utils.py
@time: 2018/11/7 16:43
"""
__author__ = '🍊 Adonis Wu 🍊'
import re

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_TOKEN = '[START]'
STOP_TOKEN = '[STOP]'
reg = "([a-zA-Z]+)([?!.]+)([a-zA-Z]+)"

from nltk.tokenize import WordPunctTokenizer


def word_tokenizer(sentence: str):
    """
    :param sentence: original sentence
    :return: words in sentence
    """
    words = WordPunctTokenizer().tokenize(sentence)
    return words


def segment_predict(sentence: str, vocabs: list, encode_max_length: int):
    """
    :param sentence: original sentence
    :param vocabs: vocab list
    :param encode_max_length: max encode length limit
    :return: encode_ids, decode_ids
    """
    #   content
    words = get_words(sentence)

    end_id = vocabs.index(STOP_TOKEN)
    pad_id = vocabs.index(PAD_TOKEN)

    encode_ids = word2id(words, vocabs)

    encode_ids = pad(encode_ids, encode_max_length, pad_id, end_id)
    return encode_ids


def segment_train(sentence: str, title: str, vocabs: list, encode_max_length: int, decode_max_length: int):
    """
    :param sentence: original sentence
    :param title: title sentence
    :param vocabs: vocab list
    :param encode_max_length: max encode length limit
    :param decode_max_length: max decode length limit
    :return: encode_ids, decode_ids, decode_target
    """
    #   content
    sen_words = get_words(sentence)
    end_id = vocabs.index(STOP_TOKEN)
    pad_id = vocabs.index(PAD_TOKEN)
    encode_ids = word2id(sen_words, vocabs)
    encode_ids = pad(encode_ids, encode_max_length, pad_id, end_id)

    #   title
    title_words = get_words(title)
    title_ids = word2id(title_words, vocabs)
    decode_ids = title_ids[:]

    decode_ids = pad(decode_ids, decode_max_length, pad_id, end_id)
    assert len(decode_ids) == decode_max_length
    return encode_ids, decode_ids


def get_words(sentence: str):
    """
    :param sentence: original sentence
    :return: use nltk segment words list
    """
    words = word_tokenizer(sentence)
    new_words = list()
    for word in words:
        #   especially for i like you.You like me. the word  you.You will be segment by this reg
        finds = re.findall(reg, word)
        if len(finds) == 0:
            new_words.append(word)
        else:
            for each in finds:
                for w in each:
                    new_words.append(w)
    return new_words


def word2id(words: list, vocabs: list):
    """
    :param words: word list
    :param vocabs: vocab list
    :return: word -> id list
    """
    ids = list()
    for word in words:
        if word in vocabs:
            ids.append(vocabs.index(word))
        else:
            ids.append(vocabs.index(UNKNOWN_TOKEN))
    return ids


def id2word(ids: list, vocabs: list):
    def to_word(limit_ids):
        words = [vocabs[_id] for _id in limit_ids]
        return words

    try:
        index = ids.index(STOP_TOKEN)
        ids = ids[:index]
        return to_word(ids)
    except ValueError:
        #   no eos_id in the list
        ids = ids[:]
        return to_word(ids)


def pad(ids: list, max_length: int, pad_id: int, end_id: int):
    """
    :param ids: original ids
    :param max_length: max length
    :param pad_id: pad id
    :param end_id: end id
    :return: padded ids
    """
    if len(ids) >= max_length:
        ids = ids[:(max_length - 1)] + [end_id]
    else:
        ids = ids + [pad_id] * ((max_length - 1) - len(ids)) + [end_id]
    return ids


def store_vocab():
    vocabs_dict = dict()

    def set_vocab(lists):
        for word in lists:
            if word in vocabs_dict.keys():
                vocabs_dict[str(word)] += 1
            else:
                vocabs_dict[str(word)] = 1

    import codecs, json, glob
    path = '/data1/ado/ado-title/metadata/train/*.train.*'
    files = glob.glob(path)
    for f in files:
        print('current file is {}'.format(f))
        data = codecs.open(f, 'r')
        for i, line in enumerate(data):
            try:
                j_line = json.loads(line)
                content = j_line['content']
                content = get_words(content)
                title = j_line['title']
                title = get_words(title)

                content.extend(title)
            except json.decoder.JSONDecodeError:
                print('error')
                continue

            set_vocab(content)

    print('vocab size is {}'.format(len(vocabs_dict)))

    store_path = '/data1/ado/ado-title/metadata/nltk_vocab.txt'

    writer = codecs.open(store_path, 'w')

    sorted_vocabs = sorted(vocabs_dict.items(), key=lambda d: d[1], reverse=True)
    index = 0
    for word, count in sorted_vocabs:
        if index == 100000:
            break
        writer.writelines(str(word) + '\t' + str(count) + '\n')
        index += 1

    print(' all finished')


if __name__ == '__main__':
    store_vocab()
