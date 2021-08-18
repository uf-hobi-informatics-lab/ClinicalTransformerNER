# -*- coding: utf-8 -*-

"""
To run this file, please use:

python <gold standard folder> <system output folder>

e.g.: python gold_annotations system_annotations

Please note that you must use Python 3 to get the correct results with this script

"""


import argparse
import glob
import os
from collections import defaultdict


class ClinicalConcept(object):
    """Named Entity Tag class."""

    def __init__(self, tid, start, end, ttype, text=''):
        """Init."""
        self.tid = str(tid).strip()
        self.start = int(start)
        self.end = int(end)
        self.text = str(text).strip()
        self.ttype = str(ttype).strip()

    def span_matches(self, other, mode='strict'):
        """Return whether the current tag overlaps with the one provided."""
        assert mode in ('strict', 'lenient')
        if mode == 'strict':
            if self.start == other.start and self.end == other.end:
                return True
        else:   # lenient
            if (self.end > other.start and self.start < other.end) or \
               (self.start < other.end and other.start < self.end):
                return True
        return False

    def equals(self, other, mode='strict'):
        """Return whether the current tag is equal to the one provided."""
        assert mode in ('strict', 'lenient')
        return other.ttype == self.ttype and self.span_matches(other, mode)

    def __str__(self):
        """String representation."""
        return '{}\t{}\t({}:{})'.format(self.ttype, self.text, self.start, self.end)


class Relation(object):
    """Relation class."""

    def __init__(self, rid, arg1, arg2, rtype):
        """Init."""
        assert isinstance(arg1, ClinicalConcept)
        assert isinstance(arg2, ClinicalConcept)
        self.rid = str(rid).strip()
        self.arg1 = arg1
        self.arg2 = arg2
        self.rtype = str(rtype).strip()

    def equals(self, other, mode='strict'):
        """Return whether the current tag is equal to the one provided."""
        assert mode in ('strict', 'lenient')
        if self.arg1.equals(other.arg1, mode) and \
                self.arg2.equals(other.arg2, mode) and \
                self.rtype == other.rtype:
            return True
        return False

    def __str__(self):
        """String representation."""
        return '{} ({}->{})'.format(self.rtype, self.arg1.ttype,
                                    self.arg2.ttype)


class RecordTrack2(object):
    """Record for Track 2 class."""

    def __init__(self, file_path, tags_to_exclude=None):
        """Initialize."""
        self.path = os.path.abspath(file_path)
        self.basename = os.path.basename(self.path)
        self.tags_to_exclude = tags_to_exclude
        self.annotations = self._get_annotations()

    @property
    def tags(self):
        return self.annotations['tags']

    @property
    def relations(self):
        return self.annotations['relations']

    def _get_annotations(self):
        """Return a dictionary with all the annotations in the .ann file."""
        annotations = defaultdict(dict)
        with open(self.path) as annotation_file:
            lines = annotation_file.readlines()
            for line_num, line in enumerate(lines):
                if line.strip().startswith('T'):
                    try:
                        tag_id, tag_m, tag_text = line.strip().split('\t')
                    except ValueError:
                        print(self.path, line)
                    if len(tag_m.split(' ')) == 3:
                        tag_type, tag_start, tag_end = tag_m.split(' ')
                    elif len(tag_m.split(' ')) == 4:
                        tag_type, tag_start, _, tag_end = tag_m.split(' ')
                    elif len(tag_m.split(' ')) == 5:
                        tag_type, tag_start, _, _, tag_end = tag_m.split(' ')
                    else:
                        print(self.path)
                        print(line)
                    tag_start, tag_end = int(tag_start), int(tag_end)
                    # add exclude tag function
                    if self.tags_to_exclude and tag_type.lower() in self.tags_to_exclude:
                        continue
                    annotations['tags'][tag_id] = ClinicalConcept(tag_id, tag_start, tag_end, tag_type, tag_text)
            for line_num, line in enumerate(lines):
                if line.strip().startswith('R'):
                    rel_id, rel_m = line.strip().split('\t')
                    rel_type, rel_arg1, rel_arg2 = rel_m.split(' ')
                    rel_arg1 = rel_arg1.split(':')[1]
                    rel_arg2 = rel_arg2.split(':')[1]
                    arg1 = annotations['tags'][rel_arg1]
                    arg2 = annotations['tags'][rel_arg2]
                    annotations['relations'][rel_id] = Relation(rel_id, arg1, arg2, rel_type)
        return annotations

    def _get_text(self):
        """Return the text in the corresponding txt file."""
        path = self.path.replace('.ann', '.txt')
        with open(path) as text_file:
            text = text_file.read()
        return text

    def search_by_id(self, key):
        """Search by id among both tags and relations."""
        try:
            return self.annotations['tags'][key]
        except KeyError():
            try:
                return self.annotations['relations'][key]
            except KeyError():
                return None


class Measures(object):
    """Abstract methods and var to evaluate."""

    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        """Initizialize."""
        assert type(tp) == int
        assert type(tn) == int
        assert type(fp) == int
        assert type(fn) == int
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def precision(self):
        """Compute Precision score."""
        try:
            return self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            return 0.0

    def recall(self):
        """Compute Recall score."""
        try:
            return self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            return 0.0

    def f_score(self, beta=1):
        """Compute F1-measure score."""
        assert beta > 0.
        try:
            num = (1 + beta**2) * (self.precision() * self.recall())
            den = beta**2 * (self.precision() + self.recall())
            return num / den
        except ZeroDivisionError:
            return 0.0

    def f1(self):
        """Compute the F1-score (beta=1)."""
        return self.f_score(beta=1)

    def specificity(self):
        """Compute Specificity score."""
        try:
            return self.tn / (self.fp + self.tn)
        except ZeroDivisionError:
            return 0.0

    def sensitivity(self):
        """Compute Sensitivity score."""
        return self.recall()

    def auc(self):
        """Compute AUC score."""
        return (self.sensitivity() + self.specificity()) / 2


class SingleEvaluator(object):
    """Evaluate two single files."""

    def __init__(self, doc1, doc2, track, mode='strict', key=None, verbose=False):
        """Initialize."""
        assert isinstance(doc1, RecordTrack2)
        assert isinstance(doc2, RecordTrack2)
        assert mode in ('strict', 'lenient')
        assert doc1.basename == doc2.basename
        self.scores = {'tags': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
                       'relations': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}}
        self.doc1 = doc1
        self.doc2 = doc2
        if key:
            gol = [t for t in doc1.tags.values() if key == t.ttype]
            sys = [t for t in doc2.tags.values() if key == t.ttype]
            sys_check = [t for t in doc2.tags.values() if key == t.ttype]
        else:
            gol = [t for t in doc1.tags.values()]
            sys = [t for t in doc2.tags.values()]
            sys_check = [t for t in doc2.tags.values()]
        #pare down matches -- if multiple system tags overlap with only one
        #gold standard tag, only keep one sys tag
        gol_matched = []
        for s in sys:
            for g in gol:
                if g.equals(s, mode):
                    if g not in gol_matched:
                        gol_matched.append(g)
                    else:
                        if s in sys_check:
                            sys_check.remove(s)

        sys = sys_check
        #now evaluate
        self.scores['tags']['tp'] = len({s.tid for s in sys for g in gol if g.equals(s, mode)})
        self.scores['tags']['fp'] = len({s.tid for s in sys}) - self.scores['tags']['tp']
        self.scores['tags']['fn'] = len({g for g in gol}) - self.scores['tags']['tp']
        self.scores['tags']['tn'] = 0

        if verbose and track == 2:
            tps = {s for s in sys for g in gol if g.equals(s, mode)}
            fps = set(sys) - tps
            fns = set()
            for g in gol:
                if not len([s for s in sys if s.equals(g, mode)]):
                    fns.add(g)
            for e in fps:
                print('FP: ' + str(e))
            for e in fns:
                print('FN:' + str(e))
        if track == 2:
            if key:
                gol = [r for r in doc1.relations.values() if r.rtype == key]
                sys = [r for r in doc2.relations.values() if r.rtype == key]
                sys_check = [r for r in doc2.relations.values() if r.rtype == key]
            else:
                gol = [r for r in doc1.relations.values()]
                sys = [r for r in doc2.relations.values()]
                sys_check = [r for r in doc2.relations.values()]

            #pare down matches -- if multiple system tags overlap with only one
            #gold standard tag, only keep one sys tag
            gol_matched = []
            for s in sys:
                for g in gol:
                    if g.equals(s,mode):
                        if g not in gol_matched:
                            gol_matched.append(g)
                        else:
                            if s in sys_check:
                                sys_check.remove(s)
            sys = sys_check
            #now evaluate
            self.scores['relations']['tp'] = len({s for s in sys for g in gol if g.equals(s, mode)})
            self.scores['relations']['fp'] = len({s for s in sys}) - self.scores['relations']['tp']
            self.scores['relations']['fn'] = len({g for g in gol}) - self.scores['relations']['tp']
            self.scores['relations']['tn'] = 0
            if verbose:
                tps = {s for s in sys for g in gol if g.equals(s, mode)}
                fps = set(sys) - tps
                fns = set()
                for g in gol:
                    if not len([s for s in sys if s.equals(g, mode)]):
                        fns.add(g)
                for e in fps:
                    print('FP: ' + str(e))
                for e in fns:
                    print('FN:' + str(e))

    def __str__(self):
        return str(self.__dict__)


class MultipleEvaluator(object):
    """Evaluate two sets of files."""

    def __init__(self, corpora, tags, relations, tag_type=None, mode='strict', verbose=False):
        """Initialize."""
        assert isinstance(corpora, Corpora)
        assert mode in ('strict', 'lenient')
        self.scores = None
        self.tags = tags
        self.relations = relations
        self.track2(corpora, tag_type, mode, verbose)

    def track2(self, corpora, tag_type=None, mode='strict', verbose=False):
        """Compute measures for Track 2."""
        self.scores = {'tags': {'tp': 0,
                                'fp': 0,
                                'fn': 0,
                                'tn': 0,
                                'micro': {'precision': 0,
                                          'recall': 0,
                                          'f1': 0},
                                'macro': {'precision': 0,
                                          'recall': 0,
                                          'f1': 0}},
                       'relations': {'tp': 0,
                                     'fp': 0,
                                     'fn': 0,
                                     'tn': 0,
                                     'micro': {'precision': 0,
                                               'recall': 0,
                                               'f1': 0},
                                     'macro': {'precision': 0,
                                               'recall': 0,
                                               'f1': 0}}}

        for g, s in corpora.docs:
            evaluator = SingleEvaluator(g, s, 2, mode, tag_type, verbose=verbose)
            for target in ('tags', 'relations'):
                for score in ('tp', 'fp', 'fn'):
                    self.scores[target][score] += evaluator.scores[target][score]
                measures = Measures(tp=evaluator.scores[target]['tp'],
                                    fp=evaluator.scores[target]['fp'],
                                    fn=evaluator.scores[target]['fn'],
                                    tn=evaluator.scores[target]['tn'])
                for score in ('precision', 'recall', 'f1'):
                    fn = getattr(measures, score)
                    self.scores[target]['macro'][score] += fn()

        for target in ('tags', 'relations'):
            # Normalization
            for key in self.scores[target]['macro'].keys():
                self.scores[target]['macro'][key] = self.scores[target]['macro'][key] / len(corpora.docs)

            measures = Measures(tp=self.scores[target]['tp'],
                                fp=self.scores[target]['fp'],
                                fn=self.scores[target]['fn'],
                                tn=self.scores[target]['tn'])
            for key in self.scores[target]['micro'].keys():
                fn = getattr(measures, key)
                self.scores[target]['micro'][key] = fn()

    def __str__(self):
        return str(self.__dict__)


def evaluate(corpora, annotations, mode='strict', verbose=False):
    """Run the evaluation by considering only files in the two folders."""
    assert mode in ('strict', 'lenient')
    # tags and relations annotated in gold standard
    gs_tags, gs_rels = annotations
    evaluator_s = MultipleEvaluator(corpora, gs_tags, gs_rels, verbose=verbose)
    evaluator_l = MultipleEvaluator(corpora, gs_tags, gs_rels, mode='lenient', verbose=verbose)
    print('{:*^70}'.format(' Information Extraction Results '))
    print('{:20}  {:-^22}    {:-^22}'.format('', ' strict ', ' lenient '))
    print('{:20}  {:6}  {:6}  {:6}    {:6}  {:6}  {:6}'.format('', 'Prec.',
                                                               'Rec.',
                                                               'F(b=1)',
                                                               'Prec.',
                                                               'Rec.',
                                                               'F(b=1)'))
    for tag in evaluator_s.tags:
        evaluator_tag_s = MultipleEvaluator(corpora, gs_tags, gs_rels, tag, verbose=verbose)
        evaluator_tag_l = MultipleEvaluator(corpora, gs_tags, gs_rels, tag, mode='lenient', verbose=verbose)
        print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
            tag.capitalize(),
            evaluator_tag_s.scores['tags']['micro']['precision'],
            evaluator_tag_s.scores['tags']['micro']['recall'],
            evaluator_tag_s.scores['tags']['micro']['f1'],
            evaluator_tag_l.scores['tags']['micro']['precision'],
            evaluator_tag_l.scores['tags']['micro']['recall'],
            evaluator_tag_l.scores['tags']['micro']['f1']))
    print('{:>20}  {:-^48}'.format('', ''))
    print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
        'Overall (micro)',
        evaluator_s.scores['tags']['micro']['precision'],
        evaluator_s.scores['tags']['micro']['recall'],
        evaluator_s.scores['tags']['micro']['f1'],
        evaluator_l.scores['tags']['micro']['precision'],
        evaluator_l.scores['tags']['micro']['recall'],
        evaluator_l.scores['tags']['micro']['f1']))
    print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
        'Overall (macro)',
        evaluator_s.scores['tags']['macro']['precision'],
        evaluator_s.scores['tags']['macro']['recall'],
        evaluator_s.scores['tags']['macro']['f1'],
        evaluator_l.scores['tags']['macro']['precision'],
        evaluator_l.scores['tags']['macro']['recall'],
        evaluator_l.scores['tags']['macro']['f1']))
    print()

    print('{:*^70}'.format(' RELATIONS '))
    for rel in evaluator_s.relations:
        evaluator_tag_s = MultipleEvaluator(corpora, gs_tags, gs_rels, rel, mode='strict', verbose=verbose)
        evaluator_tag_l = MultipleEvaluator(corpora, gs_tags, gs_rels, rel, mode='lenient', verbose=verbose)
        print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
            rel,
            evaluator_tag_s.scores['relations']['micro']['precision'],
            evaluator_tag_s.scores['relations']['micro']['recall'],
            evaluator_tag_s.scores['relations']['micro']['f1'],
            evaluator_tag_l.scores['relations']['micro']['precision'],
            evaluator_tag_l.scores['relations']['micro']['recall'],
            evaluator_tag_l.scores['relations']['micro']['f1']))
    print('{:>20}  {:-^48}'.format('', ''))
    print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
        'Overall (micro)',
        evaluator_s.scores['relations']['micro']['precision'],
        evaluator_s.scores['relations']['micro']['recall'],
        evaluator_s.scores['relations']['micro']['f1'],
        evaluator_l.scores['relations']['micro']['precision'],
        evaluator_l.scores['relations']['micro']['recall'],
        evaluator_l.scores['relations']['micro']['f1']))
    print('{:>20}  {:<5.4f}  {:<5.4f}  {:<5.4f}    {:<5.4f}  {:<5.4f}  {:<5.4f}'.format(
        'Overall (macro)',
        evaluator_s.scores['relations']['macro']['precision'],
        evaluator_s.scores['relations']['macro']['recall'],
        evaluator_s.scores['relations']['macro']['f1'],
        evaluator_l.scores['relations']['macro']['precision'],
        evaluator_l.scores['relations']['macro']['recall'],
        evaluator_l.scores['relations']['macro']['f1']))
    print()
    print('{:20}{:^48}'.format('', '  {} files found  '.format(len(corpora.docs))))


class Corpora(object):
    def __init__(self, folder1, folder2, tags_to_exclude=None):
        file_ext = '*.ann'
        self.folder1 = folder1
        self.folder2 = folder2
        files1 = set([os.path.basename(f) for f in glob.glob(os.path.join(folder1, file_ext))])
        files2 = set([os.path.basename(f) for f in glob.glob(os.path.join(folder2, file_ext))])
        common_files = files1 & files2     # intersection
        if not common_files:
            print('ERROR: None of the files match.')
        else:
            if files1 - common_files:
                print('Files skipped in {}:'.format(self.folder1))
                print(', '.join(sorted(list(files1 - common_files))))
            if files2 - common_files:
                print('Files skipped in {}:'.format(self.folder2))
                print(', '.join(sorted(list(files2 - common_files))))
        self.docs = []
        for file in common_files:
            g = RecordTrack2(os.path.join(self.folder1, file), tags_to_exclude)
            s = RecordTrack2(os.path.join(self.folder2, file), tags_to_exclude)
            self.docs.append((g, s))

    def get_annotations(self):
        gs_tags = set()
        gs_rels = set()
        for each in self.docs:
            g = each[0]
            for _, ann in g.tags.items():
                gs_tags.add(ann.ttype)
            for _, rel in g.relations.items():
                gs_rels.add(rel.rtype)
        return sorted(gs_tags, key=lambda x: len(x)), sorted(gs_rels, key=lambda x: len(x))


def eval_files(f1, f2, verbose, tags_to_exclude=None):
    """Where the magic begins."""
    corpora = Corpora(f1, f2, tags_to_exclude)
    annotations = corpora.get_annotations()
    if corpora.docs:
        evaluate(corpora, annotations, verbose=verbose)


def load_exclude_tags(fn):
    tag_list = []
    with open(fn, "r") as f:
        for line in f.readlines():
            tag_list.append(line.strip().lower())
    return set(tag_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='n2c2: Evaluation script for Track 2')
    parser.add_argument('--f1', help='First data folder path (gold)')
    parser.add_argument('--f2', help='Second data folder path (system)')
    parser.add_argument('-v', '--verbose', default=0, help='verbosity')
    parser.add_argument('-e', '--exclude', default=None, type=str, help='a file with all tags you do not want to eval')
    args = parser.parse_args()
    exclude_tags = None
    if args.exclude and os.path.exists(args.exclude):
        exclude_tags = load_exclude_tags(args.exclude)
    eval_files(os.path.abspath(args.f1), os.path.abspath(args.f2), int(args.verbose), exclude_tags)
