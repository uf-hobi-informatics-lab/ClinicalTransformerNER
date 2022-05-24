#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: nested_inference.sh
#

REPO_PATH=/home/alexgre/projects/2022n2c2/2022n2c2_track2/mrc_ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_SIGN=sdoh_trigger

DATA_DIR=/home/alexgre/projects/2022n2c2/2022n2c2_track2/data/mrc_trigger/

FILE=gatortron-syn-345m_deid_vocab
BERT_DIR=/home/alexgre/projects/transformer_pretrained_models/${FILE}

MAX_LEN=512

predict_output=/home/alexgre/projects/2022n2c2/2022n2c2_track2/exp/pred_${FILE}.json

OUTPUT_BASE=/home/alexgre/projects/2022n2c2/2022n2c2_track2/exp/model

model_root=${OUTPUT_BASE}/gatortron-syn-345m_deid_vocab_4_2e-5_30

MODEL_CKPT=${model_root}/epoch=6.ckpt

HPARAMS_FILE=${model_root}/lightning_logs/version_0/hparams.yaml


python3 ${REPO_PATH}/inference/mrc_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--output_fn ${predict_output} \
--dataset_sign ${DATA_SIGN}