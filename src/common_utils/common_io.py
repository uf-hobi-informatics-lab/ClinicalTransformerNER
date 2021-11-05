# -*- coding: utf-8 -*-

import json
import pickle as pkl


def read_from_file(ifn):
    with open(ifn, "r") as f:
        text = f.read()
    return text


def write_to_file(text, ofn):
    with open(ofn, "w") as f:
        f.write(text)
    return True


def pkl_load(ifn):
    with open(ifn, "rb") as f:
        pdata = pkl.load(f)
    return pdata


def pkl_dump(pdata, ofn):
    with open(ofn, "wb") as f:
        pkl.dump(pdata, f)
    return True


def json_load(ifn):
    with open(ifn, "r") as f:
        jdata = json.load(f)
    return jdata


def json_dump(jdata, ofn):
    with open(ofn, "w") as f:
        json.dump(jdata, f)
    return True


def load_bio_file_into_sents(bio_file, word_sep=" ", do_lower=False):
    bio_text = read_from_file(bio_file)
    bio_text = bio_text.strip()
    if do_lower:
        bio_text = bio_text.lower()

    new_sents = []
    sents = bio_text.split("\n\n")

    for sent in sents:
        new_sent = []
        words = sent.split("\n")
        for word in words:
            new_word = word.split(word_sep)
            new_sent.append(new_word)
        new_sents.append(new_sent)

    return new_sents


def output_bio(bio_data, output_file, sep=" "):
    with open(output_file, "w") as f:
        for sent in bio_data:
            for word in sent:
                line = sep.join(word)
                f.write(line)
                f.write("\n")
            f.write("\n")
