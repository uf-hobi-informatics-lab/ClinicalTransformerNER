# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
The script provide a tool to convert BIO formatted file to either Brat or BioC format
The script also provide a tool to merge several brat or BioC formatted files into one file by concatenating all the unique entities.

The pre-request is BIO data must have offset information
"""

import shutil
import traceback
from pathlib import Path
from collections import defaultdict

from ..common_utils.common_io import load_bio_file_into_sents, read_from_file, json_load, pkl_load, write_to_file

BRAT_TEMPLATE = "{}\t{} {} {}\t{}"
BIOC_TEMPLATE = """
      <annotation id="{a}">
        <infon key="type">{e}</infon>
        <location length="{d}" offset="{c}"/>
        <text>{b}</text>
      </annotation>\n
"""
BIOC_HEADER = """
<?xml version='1.0' encoding='utf-8' standalone='yes'?>
<collection>
  <source></source>
  <date></date>
  <key></key>
  <document>
    <id>{}</id>
    <passage>
      <offset>0</offset>
"""
BIOC_END = """
    </passage>
  </document>
</collection>
"""


def _print_info(predictions):
    # predict first three predictions
    i = 0
    for each in predictions:
        if each["entities"]:
            i += 1
            print(each)
            print()
        if i > 3:
            break


def biaffine2bio(raw_bio_file, biaffine_output_file, formatted_output_dir):
    p_output = Path(formatted_output_dir)
    p_output.mkdir(exist_ok=True, parents=True)
    output_fn = p_output / "predicted_bio.txt"

    sents = load_bio_file_into_sents(raw_bio_file)
    nsents = []
    for sent in sents:
        nsent = []
        for word in sent:
            nsent.append(word[0])
        nsents.append(nsent)

    predictions = json_load(biaffine_output_file)
    _print_info(predictions)

    assert len(predictions) == len(nsents), f"expect same predictions and mappings but get " \
                                              f"{len(predictions)} predictions and {len(nsents)} mappings"

    labeled_sents = []
    for pred, sent in zip(predictions, nsents):
        ens = pred['entities']
        labels = ["O"] * len(sent)
        for en in ens:
            en_type = en[0]
            s, e = en[1:3]
            if s >= e:
                continue

            for i in range(s, e):
                if (i - s) == 0:
                    labels[i] = "B-" + en_type
                else:
                    labels[i] = "I-" + en_type

        labeled_sents.append("\n".join([f"{w} {l}" for w, l in zip(sent, labels)]))

    write_to_file("\n\n".join(labeled_sents), output_fn)


def biaffine2brat(raw_text_dir, biaffine_output_file, mapping_file, formatted_output_dir, do_copy):
    p = Path(raw_text_dir)
    p_output = Path(formatted_output_dir)
    p_output.mkdir(exist_ok=True, parents=True)

    predictions = json_load(biaffine_output_file)
    _print_info(predictions)

    mappings = pkl_load(mapping_file)

    assert len(predictions) == len(mappings), f"expect same predictions and mappings but get " \
                                              f"{len(predictions)} predictions and {len(mappings)} mappings"

    nid = ""
    ntext = ""
    idx = 1
    res_dict = defaultdict(list)

    for pred, sent_map in zip(predictions, mappings):
        ens = pred['entities']
        for en in ens:
            en_type = en[0]
            s, e = en[1:3]
            if s >= e:
                continue

            words = sent_map[s: e]
            note_id = words[0][-1]

            if nid != note_id:
                nid = note_id
                ntext = read_from_file(p/f"{nid}.txt")
                idx = 1

            org_s, org_e = words[0][1]
            for w in words[1:]:
                _, org_e = w[1]

            en_text = ntext[org_s: org_e].replace("\n", " ")

            res_dict[nid].append(BRAT_TEMPLATE.format(idx, en_type, org_s, org_e, en_text))
            idx += 1

    for fid, brats in res_dict.items():
        write_to_file("\n".join(brats), p_output/f"{fid}.ann")

    if do_copy:
        for fn in p.glob("*.txt"):
            shutil.copyfile(fn,  p_output/fn.name)


def __prepare_path(text_dir, input_dir, output_dir, write_output):
    t_input = Path(text_dir)
    p_input = Path(input_dir)
    p_output = Path(output_dir)
    if write_output:
        p_output.mkdir(parents=True, exist_ok=True)

    return t_input, p_input, p_output


def tag2entity(sents):
    entities = []
    for sent in sents:
        term, start, end, sem_tag, prev_tag = [], None, None, None, "O"
        for text, w_s, w_e, w_a_s, w_a_e, predict_tag in sent:
            if predict_tag == "O":
                if prev_tag != "O":
                    entities.append((" ".join(term), start, end, sem_tag))
                    term, start, end, sem_tag = [], None, None, None
            else:
                boundary, ttag = predict_tag.split("-")
                if boundary == "B":
                    if prev_tag != "O":
                        entities.append((" ".join(term), start, end, sem_tag))
                        term, start, end, sem_tag = [], None, None, None
                    term.append(text)
                    start, end, sem_tag = w_s, w_e, ttag
                elif boundary == "I":
                    if sem_tag == ttag:
                        term.append(text)
                        end = w_e
                    else:
                        if prev_tag != "O":
                            entities.append((" ".join(term), start, end, sem_tag))
                            term, start, end, sem_tag = [], None, None, None
                        term.append(text)
                        start, end, sem_tag = w_s, w_e, ttag
                else:
                    raise ValueError('The BIO scheme only support B, I but get {}-{} in {}'.format(boundary, ttag, sent))
            prev_tag = predict_tag

        if term:
            entities.append((" ".join(term), start, end, sem_tag))

    return entities

def bio2output(text_dir, input_dir, output_dir, output_template, do_copy_text, file_suffix='ann', labeled_bio_tup_lst={}, use_bio=True, write_output=True, return_dict=None):
    """
    we expect the input as a directory of all bio files end with .txt suffix
    we expect the each bio file contain the offset info (start; end position of each words) and tag info;
    original words are not required
    convert the bio formatted files to brat formatted .ann file
    the output directory will not contain the .txt file
    """
    t_input, p_input, p_output = __prepare_path(text_dir, input_dir, output_dir, write_output)
    output_dict = dict()
    for ifn in (p_input.glob("*.txt") if use_bio else labeled_bio_tup_lst.keys()):
        try:
            ifn_stem = (ifn.stem if use_bio else ifn).split(".")[0]
            doc_text_file = t_input / "{}.txt".format(ifn_stem)
            ofn = p_output / "{}.{}".format(ifn_stem, file_suffix)
            sents = labeled_bio_tup_lst.get(ifn,{}).get('sents', None)
            sents = sents if sents is not None else load_bio_file_into_sents(ifn, do_lower=False)
            doc_text = labeled_bio_tup_lst.get(ifn,{}).get('raw_text', None)
            doc_text = doc_text if doc_text is not None else read_from_file(doc_text_file)
            if len(sents[0][0][0]) == 0: 
                # not len(''.join(sum(sum(sents,[]),[]))):
                continue
            entities = tag2entity(sents)
            output_entities = []
            for idx, entity in enumerate(entities):
                ann_text, offset_s, offset_e, sem_tag = entity
                offset_s, offset_e = int(offset_s), int(offset_e)
                # we need to use original text not the ann text here
                # you can use ann_text for debugging
                raw_entity_text = doc_text[offset_s:offset_e]

                if "\n" in raw_entity_text:
                    idx = raw_entity_text.index("\n")
                    offset_s = "{} {};{}".format(offset_s, offset_s+idx, offset_s+idx+1)
                    raw_entity_text = raw_entity_text.replace("\n", " ")

                if file_suffix == "ann":
                    formatted_output = output_template.format("T{}".format(idx+1), sem_tag, offset_s, offset_e, raw_entity_text)
                elif file_suffix == "xml":
                    formatted_output = output_template.format(a=idx+1, b=raw_entity_text, c=offset_s, d=offset_e-offset_s, e=sem_tag)
                else:
                    formatted_output = None
                    print('formatted output is None due to unknown formatter code')

                output_entities.append(formatted_output)

            if return_dict is not None:
                return_dict[ifn] = [('T{}'.format(i+1),x[0],int(x[1]),int(x[2]),x[3]) for i, x in enumerate(entities)]
                
            if do_copy_text:
                new_text_file = p_output / "{}.txt".format(ifn_stem)
                shutil.copy2(doc_text_file.as_posix(), new_text_file.as_posix())

            if write_output:
                with open(ofn, "w") as f:
                    formatted_output = "\n".join(output_entities)
                    if file_suffix == "xml":
                        formatted_output = BIOC_HEADER.format(ifn.stem) + formatted_output + BIOC_END
                    f.write(formatted_output)
                    f.write("\n")
        except Exception as ex:
            traceback.print_exc()

def main(text_dir=None, input_bio_dir=None, output_dir=None, formatter=1, do_copy_text=True, labeled_bio_tup_lst={}, use_bio=True):
    if formatter == 1:
        bio2output(text_dir, input_bio_dir, output_dir, BRAT_TEMPLATE, do_copy_text, file_suffix="ann", labeled_bio_tup_lst=labeled_bio_tup_lst, use_bio=use_bio)
    elif formatter == 2:
        bio2output(text_dir, input_bio_dir, output_dir, BIOC_TEMPLATE, do_copy_text, file_suffix='xml', labeled_bio_tup_lst=labeled_bio_tup_lst, use_bio=use_bio)
    else:
        raise RuntimeError("Only support formatter as 1 and 2 but get {}; see help for more information.".format(formatter))
