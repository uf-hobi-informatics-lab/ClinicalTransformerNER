# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict
from itertools import chain
from math import pow
from pathlib import Path


def read_from_file(ifn):
    with open(ifn, "r") as f:
        text = f.read()
    return text


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


class PRF:
    def __init__(self):
        self.true = 0
        self.false = 0

    def add_true_case(self):
        self.true += 1

    def add_false_case(self):
        self.false += 1

    def get_true_false_counts(self):
        return self.true, self.false

    def __str__(self):
        return str(self.__dict__)


class BioEval:
    def __init__(self):
        self.acc = PRF()
        # prediction
        self.all_strict = PRF()
        self.all_relax = PRF()
        self.cat_strict = defaultdict(PRF)
        self.cat_relax = defaultdict(PRF)
        # gold standard
        self.gs_all = 0
        self.gs_cat = defaultdict(int)
        self.performance = dict()
        self.counts = dict()
        self.beta = 1
        self.label_not_for_eval = {'o'}

    def set_beta_for_f_score(self, beta):
        # self.logger.warning("Using beta={} for calculating F-score".format(beta))
        self.beta = beta

    def add_labels_not_for_eval(self, *labels):
        for each in labels:
            self.label_not_for_eval.add(each.lower())

    def __calc_prf(self, tp, fp, tp_tn):
        """
        Using this function to calculate F-beta score, beta=1 is f_score-score, set beta=2 favor recall, and set beta=0.5 favor precision.
        Using set_beta_for_f_score function to change beta value.
        """
        tp_fp = tp + fp
        pre = 1.0 * tp / tp_fp if tp_fp > 0 else 0.0
        rec = 1.0 * tp / tp_tn if tp_tn > 0 else 0.0
        beta2 = pow(self.beta, 2)
        f_beta = (1 + beta2) * pre * rec / (beta2 * pre + rec) if (pre + rec) > 0 else 0.0
        return pre, rec, f_beta

    def __measure_performance(self):
        self.performance['overall'] = dict()

        acc_true_num, acc_false_num = self.acc.get_true_false_counts()
        total_acc_num = acc_true_num + acc_false_num
        # calc acc
        overall_acc = round(1.0 * acc_true_num / total_acc_num, 4) if total_acc_num > 0 else 0.0
        self.performance['overall']['acc'] = overall_acc

        strict_true_counts, strict_false_counts = self.all_strict.get_true_false_counts()
        strict_pre, strict_rec, strict_f_score = self.__calc_prf(strict_true_counts, strict_false_counts, self.gs_all)
        self.performance['overall']['strict'] = dict()
        self.performance['overall']['strict']['precision'] = strict_pre
        self.performance['overall']['strict']['recall'] = strict_rec
        self.performance['overall']['strict']['f_score'] = strict_f_score

        relax_true_counts, relax_false_counts = self.all_relax.get_true_false_counts()
        relax_pre, relax_rec, relax_f_score = self.__calc_prf(relax_true_counts, relax_false_counts, self.gs_all)
        self.performance['overall']['relax'] = dict()
        self.performance['overall']['relax']['precision'] = relax_pre
        self.performance['overall']['relax']['recall'] = relax_rec
        self.performance['overall']['relax']['f_score'] = relax_f_score

        self.performance['category'] = dict()
        self.performance['category']['strict'] = dict()
        for k, v in self.cat_strict.items():
            self.performance['category']['strict'][k] = dict()
            stc, sfc = v.get_true_false_counts()
            p, r, f = self.__calc_prf(stc, sfc, self.gs_cat[k])
            self.performance['category']['strict'][k]['precision'] = p
            self.performance['category']['strict'][k]['recall'] = r
            self.performance['category']['strict'][k]['f_score'] = f

        self.performance['category']['relax'] = dict()
        for k, v in self.cat_relax.items():
            self.performance['category']['relax'][k] = dict()
            rtc, rfc = v.get_true_false_counts()
            p, r, f = self.__calc_prf(rtc, rfc, self.gs_cat[k])
            self.performance['category']['relax'][k]['precision'] = p
            self.performance['category']['relax'][k]['recall'] = r
            self.performance['category']['relax'][k]['f_score'] = f

    def __measure_counts(self):
        # gold standard
        self.counts['expect'] = dict()
        self.counts['expect']['overall'] = self.gs_all
        for k, v in self.gs_cat.items():
            self.counts['expect'][k] = v
        # prediction
        self.counts['prediction'] = {'strict': dict(), 'relax': dict()}
        # strict
        strict_true_counts, strict_false_counts = self.all_strict.get_true_false_counts()
        self.counts['prediction']['strict']['overall'] = dict()
        self.counts['prediction']['strict']['overall']['total'] = strict_true_counts + strict_false_counts
        self.counts['prediction']['strict']['overall']['true'] = strict_true_counts
        self.counts['prediction']['strict']['overall']['false'] = strict_false_counts
        for k, v in self.cat_strict.items():
            t, f = v.get_true_false_counts()
            self.counts['prediction']['strict'][k] = dict()
            self.counts['prediction']['strict'][k]['total'] = t + f
            self.counts['prediction']['strict'][k]['true'] = t
            self.counts['prediction']['strict'][k]['false'] = f
        # relax
        relax_true_counts, relax_false_counts = self.all_relax.get_true_false_counts()
        self.counts['prediction']['relax']['overall'] = dict()
        self.counts['prediction']['relax']['overall']['total'] = relax_true_counts + relax_false_counts
        self.counts['prediction']['relax']['overall']['true'] = relax_true_counts
        self.counts['prediction']['relax']['overall']['false'] = relax_false_counts
        for k, v in self.cat_relax.items():
            t, f = v.get_true_false_counts()
            self.counts['prediction']['relax'][k] = dict()
            self.counts['prediction']['relax'][k]['total'] = t + f
            self.counts['prediction']['relax'][k]['true'] = t
            self.counts['prediction']['relax'][k]['false'] = f

    @staticmethod
    def __strict_match(gs, pred, s_idx, e_idx, en_type):
        # check both gs and pred start with boundary 'B-'
        if gs[s_idx] != f"b-{en_type}":
            return False

        # check every token in span is the same
        for idx in range(s_idx, e_idx):
            if gs[idx] != pred[idx]:
                return False

        # check token after end in GS is not continued entity token
        if e_idx < len(gs) and gs[e_idx] == f"i-{en_type}":
            return False

        return True

    @staticmethod
    def __relax_match(gs, pred, s_idx, e_idx, en_type):
        # partial match
        for idx in range(s_idx, e_idx):
            gs_cate = gs[idx].split("-")[-1]
            pred_bound, pred_cate = pred[idx].split("-")
            if gs_cate == pred_cate == en_type:
                return True

        return False

    @staticmethod
    def __left_right_match():
        pass

    @staticmethod
    def __approximate():
        pass

    @staticmethod
    def __check_evaluated_already(gs_dict, cate, start_idx, end_idx):
        for k, v in gs_dict.items():
            c, s, e = k
            if not (e < start_idx or s > end_idx) and c == cate:
                if v == 0:
                    return True
                else:
                    gs_dict[k] -= 1
                    return False
        return False

    def __process_bio(self, gs_bio, pred_bio):
        # measure acc
        for w_idx, (gs_word, pred_word) in enumerate(zip(gs_bio, pred_bio)):
            # measure acc
            if gs_word == pred_word:
                self.acc.add_true_case()
            else:
                self.acc.add_false_case()
        # process gold standard
        llen = len(gs_bio)
        gs_dict = defaultdict(int)
        cur_idx = 0
        while cur_idx < llen:
            if gs_bio[cur_idx] in self.label_not_for_eval:
                cur_idx += 1
            else:
                start_idx = cur_idx
                end_idx = start_idx + 1
                _, cate = gs_bio[start_idx].strip().split('-')
                while end_idx < llen and gs_bio[end_idx] == f"i-{cate}":
                    end_idx += 1
                self.gs_all += 1
                self.gs_cat[cate] += 1
                gs_dict[(cate, start_idx, end_idx)] += 1
                cur_idx = end_idx
        # process predictions
        cur_idx = 0
        evaluated_gs = set()
        while cur_idx < llen:
            if pred_bio[cur_idx] in self.label_not_for_eval:
                cur_idx += 1
            else:
                start_idx = cur_idx
                end_idx = start_idx + 1
                _, cate = pred_bio[start_idx].split("-")
                while end_idx < llen and pred_bio[end_idx] == f"i-{cate}":
                    end_idx += 1
                if self.__strict_match(gs_bio, pred_bio, start_idx, end_idx, cate):
                    self.all_strict.add_true_case()
                    self.cat_strict[cate].add_true_case()
                    self.all_relax.add_true_case()
                    self.cat_relax[cate].add_true_case()
                elif self.__relax_match(gs_bio, pred_bio, start_idx, end_idx, cate):
                    if self.__check_evaluated_already(gs_dict, cate, start_idx, end_idx):
                        cur_idx = end_idx
                        continue
                    self.all_strict.add_false_case()
                    self.cat_strict[cate].add_false_case()
                    self.all_relax.add_true_case()
                    self.cat_relax[cate].add_true_case()
                else:
                    self.all_strict.add_false_case()
                    self.cat_strict[cate].add_false_case()
                    self.all_relax.add_false_case()
                    self.cat_relax[cate].add_false_case()
                cur_idx = end_idx

    def eval_file(self, gs_file, pred_file):
        # self.logger.info("processing gold standard file: {} and prediciton file: {}".format(gs_file, pred_file))
        pred_bio_sents = load_bio_file_into_sents(pred_file, do_lower=True)
        gs_bio_sents = load_bio_file_into_sents(gs_file, do_lower=True)
        # process bio data
        # check two data have same amount of sents
        assert len(gs_bio_sents) == len(pred_bio_sents), \
            "gold standard and prediction have different dimension: gs: {}; pred: {}".format(len(gs_bio_sents), len(pred_bio_sents))
        # measure performance
        for s_idx, (gs_sent, pred_sent) in enumerate(zip(gs_bio_sents, pred_bio_sents)):
            # check two sents have same No. of words
            assert len(gs_sent) == len(pred_sent), \
                "In {}th sentence, the words counts are different; gs: {}; pred: {}".format(s_idx, gs_sent, pred_sent)
            gs_sent = list(map(lambda x: x[-1], gs_sent))
            pred_sent = list(map(lambda x: x[-1], pred_sent))
            self.__process_bio(gs_sent, pred_sent)
        # get the evaluation matrix
        self.__measure_performance()
        self.__measure_counts()

    def eval_mem(self, gs, pred, do_flat=False):
        # flat sents to sent; we assume input sequences only have 1 dimension (only labels)
        if do_flat:
            # self.logger.warning('Sentences have been flatten to 1 dim.')
            gs = list(chain(*gs))
            pred = list(chain(*pred))
            gs = list(map(lambda x: x.lower(), gs))
            pred = list(map(lambda x: x.lower(), pred))
            self.__process_bio(gs, pred)
        else:
            for sidx, (gs_s, pred_s) in enumerate(zip(gs, pred)):
                gs_s = list(map(lambda x: x.lower(), gs_s))
                pred_s = list(map(lambda x: x.lower(), pred_s))
                self.__process_bio(gs_s, pred_s)

        self.__measure_performance()
        self.__measure_counts()

    def get_performance(self):
        return self.performance

    def get_counts(self):
        return self.counts

    def show_evaluation(self, digits=4):
        if len(self.performance) == 0:
            raise RuntimeError('call eval_mem() first to get the performance attribute')

        cate = self.performance['category']['strict'].keys()

        headers = ['precision', 'recall', 'f1']
        width = max(max([len(c) for c in cate]), len('overall'), digits)
        head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)

        report = head_fmt.format(u'', *headers, width=width)
        report += '\n\nstrict\n'

        row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + '\n'
        for c in cate:
            precision = self.performance['category']['strict'][c]['precision']
            recall = self.performance['category']['strict'][c]['recall']
            f1 = self.performance['category']['strict'][c]['f_score']
            report += row_fmt.format(c, *[precision, recall, f1], width=width, digits=digits)

        report += '\nrelax\n'

        for c in cate:
            precision = self.performance['category']['relax'][c]['precision']
            recall = self.performance['category']['relax'][c]['recall']
            f1 = self.performance['category']['relax'][c]['f_score']
            report += row_fmt.format(c, *[precision, recall, f1], width=width, digits=digits)

        report += '\n\noverall\n'
        report += 'acc: ' + str(self.performance['overall']['acc'])
        report += '\nstrict\n'
        report += row_fmt.format('', *[self.performance['overall']['strict']['precision'],
                                       self.performance['overall']['strict']['recall'],
                                       self.performance['overall']['strict']['f_score']], width=width, digits=digits)

        report += '\nrelax\n'
        report += row_fmt.format('', *[self.performance['overall']['relax']['precision'],
                                       self.performance['overall']['relax']['recall'],
                                       self.performance['overall']['relax']['f_score']], width=width, digits=digits)
        print(report)


def load_exclude_tags(fn):
    tag_list = []
    with open(fn, "r") as f:
        for line in f.readlines():
            tag_list.append(line.strip())
    return set(tag_list)


def main(targs):
    bio_eval = BioEval()
    if targs.exclude and Path(targs.exclude).is_file():
        exclude_tags = load_exclude_tags(targs.exclude)
        bio_eval.add_labels_not_for_eval(*exclude_tags)
    bio_eval.set_beta_for_f_score(beta=targs.beta)
    bio_eval.eval_file(gs_file=targs.file1, pred_file=targs.file2)
    bio_eval.show_evaluation()


def test():
    from pprint import pprint

    # from seqeval.metrics import classification_report
    # test within memory
    # a = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER']]
    # b = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER']]
    a = [['O', 'O', 'B-misc', 'O', 'O', 'B-misc', 'I-misc', 'I-misc', 'I-misc', 'I-misc', 'O']]
    b = [['O', 'O', 'B-misc', 'O', 'O', 'B-misc', 'I-misc', 'O', 'O', 'B-misc', 'O']]
    bio_eval = BioEval()
    bio_eval.eval_mem(a, b, do_flat=False)
    bio_eval.show_evaluation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default='main', type=str, help="run test or main")
    parser.add_argument("-f1", "--file1", required=True, help="gold standard")
    parser.add_argument("-f2", "--file2", required=True, help="prediction")
    parser.add_argument("-b", "--beta", default=1, type=int, help="f-score beta")
    parser.add_argument("-e", "--exclude", default=None, type=str, help="a file with list of tags not for evaluation")
    args = parser.parse_args()

    if args.mode == "main":
        main(args)
    else:
        test()
