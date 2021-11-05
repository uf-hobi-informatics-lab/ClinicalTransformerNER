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

from common_utils.common_io import load_bio_file_into_sents, read_from_file

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


def __prepare_path(text_dir, input_dir, output_dir):
    t_input = Path(text_dir)
    p_input = Path(input_dir)
    p_output = Path(output_dir)
    p_output.mkdir(parents=True, exist_ok=True)

    return t_input, p_input, p_output


def tag2entity(sents):
    entities = []
    for i, sent in enumerate(sents):
        term, start, end, sem_tag, prev_tag = [], None, None, None, "O"
        for j, word in enumerate(sent):
            text, w_s, w_e, w_a_s, w_a_e, predict_tag = word  # must have offset information
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


def bio2output(text_dir, input_dir, output_dir, output_template, do_copy_text, file_suffix='ann'):
    """
    we expect the input as a directory of all bio files end with .txt suffix
    we expect the each bio file contain the offset info (start; end position of each words) and tag info;
    original words are not required
    convert the bio formatted files to brat formatted .ann file
    the output directory will not contain the .txt file
    """
    t_input, p_input, p_output = __prepare_path(text_dir, input_dir, output_dir)
    for ifn in p_input.glob("*.txt"):
        try:
            ifn_stem = ifn.stem.split(".")[0]
            doc_text_file = t_input / "{}.txt".format(ifn_stem)
            ofn = p_output / "{}.{}".format(ifn_stem, file_suffix)
            sents = load_bio_file_into_sents(ifn, do_lower=False)
            doc_text = read_from_file(doc_text_file)
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

            if do_copy_text:
                new_text_file = p_output / "{}.txt".format(ifn_stem)
                shutil.copy2(doc_text_file.as_posix(), new_text_file.as_posix())

            with open(ofn, "w") as f:
                formatted_output = "\n".join(output_entities)
                if file_suffix == "xml":
                    formatted_output = BIOC_HEADER.format(ifn.stem) + formatted_output + BIOC_END
                f.write(formatted_output)
                f.write("\n")
        except Exception as ex:
            traceback.print_exc()


def main(text_dir=None, input_bio_dir=None, output_dir=None, formatter=1, do_copy_text=True):
    if formatter == 1:
        bio2output(text_dir, input_bio_dir, output_dir, BRAT_TEMPLATE, do_copy_text, file_suffix="ann")
    elif formatter == 2:
        bio2output(text_dir, input_bio_dir, output_dir, BIOC_TEMPLATE, do_copy_text, file_suffix='xml')
    else:
        raise RuntimeError("Only support formatter as 1 and 2 but get {}; see help for more information.".format(formatter))
