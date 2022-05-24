#!/usr/bin/env bash
# -*- coding: utf-8 -*-


FILE=conll03_cased_large
REPO_PATH=/home/alexgre/projects/2022n2c2/2022n2c2_track2/mrc_ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=/home/alexgre/projects/2022n2c2/2022n2c2_track2/mrc_ner/datasets/conll03/
# mimiciii-bert-large-uncased_5e_128b
BERT_DIR=/home/alexgre/projects/transformer_pretrained_models/bert-large-uncased
OUTPUT_BASE=/home/alexgre/projects/2022n2c2/2022n2c2_track2/exp

BATCH=4
GRAD_ACC=1
BERT_DROPOUT=0.1
MRC_DROPOUT=0.3
LR=1e-5
LR_MINI=5e-6
LR_SCHEDULER=linear
SPAN_WEIGHT=0.1
WARMUP=0
MAX_LEN=512
MAX_NORM=1.0
MAX_EPOCH=5
INTER_HIDDEN=2048
WEIGHT_DECAY=0.01
OPTIM=torch.adam #adamw
VAL_CHECK=0.2
PREC=16
SPAN_CAND=pred_and_gold


OUTPUT_DIR=${OUTPUT_BASE}/model/conll03
mkdir -p ${OUTPUT_DIR}


python ${REPO_PATH}/train/mrc_ner_trainer.py \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--batch_size ${BATCH} \
--gpus="1" \
--precision=${PREC} \
--progress_bar_refresh_rate 1 \
--lr ${LR} \
--val_check_interval ${VAL_CHECK} \
--accumulate_grad_batches ${GRAD_ACC} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${MRC_DROPOUT} \
--bert_dropout ${BERT_DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CAND} \
--weight_span ${SPAN_WEIGHT} \
--warmup_steps ${WARMUP} \
--distributed_backend=ddp \
--max_length ${MAX_LEN} \
--gradient_clip_val ${MAX_NORM} \
--weight_decay ${WEIGHT_DECAY} \
--optimizer ${OPTIM} \
--lr_scheduler ${LR_SCHEDULER} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN} \
--flat \
--lr_mini ${LR_MINI}

