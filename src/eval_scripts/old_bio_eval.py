###
 # <p>Title:  </p>
 # <p>Create Date: 21:23:36 01/28/18</p>
 # <p>Copyright: College of Medicine </p>
 # <p>Organization: University of Florida</p>
 # @author Yonghui Wu
 # @version 1.0
 # <p>Description: </p>
 ##
# from create_log import create_logger

from __future__ import print_function


def read_from_file(ifn):
    with open(ifn, "r") as f:
        text = f.read()
    return text


class PRF:
    def __init__(self):
        self.true=0
        self.false=0
        

class BioEval:
    def __init__(self, ifn, log_name=None):
        self.ifn=ifn
        self.acc=PRF()
        self.all_strict=PRF()
        self.all_relax=PRF()
        self.cate_strict={}
        self.cate_relax={}

        self.gold_all=0
        self.gold_cate={}
        # self.entities=[]
        self.log_name = log_name

    def eval_fn(self):
        text=read_from_file(self.ifn).strip().lower()
        secs=text.split('\n\n')
        for sec in secs:
            sec=sec.strip()
            lines=sec.split('\n')
            bio=[]
            for line in lines:
                words=line.split(None)
                #words.append(words[-1])
                bio.append(words)
            self.handle(bio)
        self.prf()

    def feed_bio(self,bio):
        self.handle(bio)

    def train_msg(self):
        stt="Entities: "
        for k, v in self.gold_cate.items():
            stt=stt+k+":"+str(v)+"  "
        if (self.acc.true+self.acc.false) > 0:
            acc=float(self.acc.true)/(self.acc.true+self.acc.false)
        else:
            acc=0.0
        if (self.all_strict.true+self.all_strict.false) > 0 and self.gold_all>0:
            pre = float(self.all_strict.true)/(self.all_strict.true+self.all_strict.false)
            rec = float(self.all_strict.true)/self.gold_all
            if pre+rec>0.0:
                f1=2*pre*rec/(pre+rec)
            else:
                f1=0.0
        else:
            pre=0.0
            rec=0.0
            f1=0.0

        #all_relex
        if (self.all_relax.true+self.all_relax.false) > 0 and self.gold_all>0:
            rpre = float(self.all_relax.true)/(self.all_relax.true+self.all_relax.false)
            rrec = float(self.all_relax.true)/self.gold_all
            if (rpre+rrec) > 0.0:
                rf1=2*rpre*rrec/(rpre+rrec)
            else:
                rf1=0.0
        else:
            rpre=0.0
            rrec=0.0
            rf1=0.0

        return([stt,f1,pre,rec,rf1,rpre,rrec,acc])

    def prf(self):
        # print "Total %s entities " % self.gold_all
        log_info = "Total %s entities " % self.gold_all + "\n"
        for k,v in self.gold_cate.items():
            # print "    %s : %s" % (k,v)
            log_info += "    %s : %s\n" % (k,v)

        acc=float(self.acc.true)/(self.acc.true+self.acc.false)
        # print "\nAccuracy : %s" % acc
        log_info += "\nAccuracy : %s\n" % acc

        pre = float(self.all_strict.true)/(self.all_strict.true+self.all_strict.false)
        rec = float(self.all_strict.true)/self.gold_all
        try:
            f1=2*pre*rec/(pre+rec)
        except ZeroDivisionError:
            f1 = 0.0

        # print "\n\nStrict score ----- "
        log_info += "\n\nStrict score ----- \n"
        # print 'precision : %s , recall : %s , f1 : %s' % (pre,rec,f1)
        log_info += 'precision : %s , recall : %s , f1 : %s\n' % (pre,rec,f1)
        # print 'find : %s , true : %s , false : %s' % (self.all_strict.true+self.all_strict.false,self.all_strict.true,self.all_strict.false)
        log_info += 'find : %s , true : %s , false : %s \n' % (self.all_strict.true+self.all_strict.false,
                                                             self.all_strict.true,self.all_strict.false)
        #all_relex
        pre = float(self.all_relax.true)/(self.all_relax.true+self.all_relax.false)
        rec = float(self.all_relax.true)/self.gold_all
        try:
            f1=2*pre*rec/(pre+rec)
        except ZeroDivisionError:
            f1 = 0.0

        # print "\nRelax score -----"
        log_info += "\nRelax score -----\n"
        # print 'precision : %s , recall : %s , f1 : %s' % (pre,rec,f1)
        log_info += 'precision : %s , recall : %s , f1 : %s\n' % (pre,rec,f1)
        # print 'find : %s , true : %s , false : %s' % (self.all_relax.true+self.all_relax.false,self.all_relax.true,self.all_relax.false)
        log_info += 'find : %s , true : %s , false : %s \n' % (self.all_relax.true+self.all_relax.false,
                                                             self.all_relax.true,self.all_relax.false)
        ##category score
        # print "\nstrict score by cate -----"
        log_info += "\nstrict score by cate -----\n"
        for k,v in self.cate_strict.items():
            pre = float(v.true)/(v.true+v.false)
            if k not in self.gold_cate:
                rec=0.0
                f1=0.0
            else:
                rec = float(v.true)/self.gold_cate[k]
                try:
                    f1 = 2 * pre * rec / (pre + rec)
                except ZeroDivisionError:
                    f1 = 0.0

            # print "Cate : %s, precision : %s , recall : %s , f1 : %s" % (k,pre,rec,f1)
            log_info += "Cate : %s, precision : %s , recall : %s , f1 : %s\n" % (k,pre,rec,f1)
            # print 'find : %s , true : %s , false : %s' % (v.true+v.false,v.true,v.false)
            log_info += 'find : %s , true : %s , false : %s\n' % (v.true+v.false,v.true,v.false)

        # print "\nrelax score by cate -----"
        log_info += "\nrelax score by cate -----\n"
        for k,v in self.cate_relax.items():
            pre = float(v.true)/(v.true+v.false)
            if k not in self.gold_cate:
                rec = 0.0
                f1 = 0.0
            else:
                rec = float(v.true)/self.gold_cate[k]
                try:
                    f1 = 2 * pre * rec / (pre + rec)
                except ZeroDivisionError:
                    f1 = 0.0

            # print "Cate : %s, precision : %s , recall : %s , f1 : %s" % (k,pre,rec,f1)
            log_info += "Cate : %s, precision : %s , recall : %s , f1 : %s\n" % (k,pre,rec,f1)
            # print 'find : %s , true : %s , false : %s' % (v.true+v.false,v.true,v.false)
            log_info += 'find : %s , true : %s , false : %s\n' % (v.true+v.false,v.true,v.false)

        print(log_info)
        # if self.log_name:
        #     logger = create_logger(self.log_name, "--evaluation--")
        #     logger.info(log_info)

    def same(self,bio,starti,endi):
        '''
        whether the ner (starti : endi) is exactly match
        '''
        flag=True
        pcate=bio[starti][-1][2:]
        if bio[starti][-2].startswith("i-"):
            cate=bio[starti][-2][2:]
            if cate != pcate:
                flag=False
            else:
                #check starti-1
                if starti -1 >= 0 and bio[starti-1][-2] == "i-"+cate or bio[starti-1][-2] == "b-"+cate:
                    flag=False
            if flag:
                for i in range(starti+1,endi):
                    if bio[i][-2] != "i-"+cate:
                        flag=False
            if flag:# check endi
                if endi < len(bio) and bio[endi][-2] == "i-"+cate:
                    flag=False
        elif bio[starti][-2].startswith("b-"):
            cate=bio[starti][-2][2:]
            if cate != pcate:
                flag=False
            # do not need check starti -1
            if flag:
                for i in range(starti+1,endi):
                    if bio[i][-2] != "i-"+cate:
                        flag=False
            if flag:# check endi
                if endi < len(bio) and bio[endi][-2] == "i-"+cate:
                    flag=False
        else:
            flag=False
            
        return flag

    def overlap(self,bio,starti,endi):
        flag=False
        for i in range(starti,endi):
            if len(bio[i][-2])> 2 and bio[i][-1][2:] == bio[i][-2][2:]:
                flag=True
                break
        return flag

    def add_tp_strict(self,cate):
        self.all_strict.true=self.all_strict.true+1
        self.all_relax.true=self.all_relax.true+1
        if cate not in self.cate_strict:
            self.cate_strict[cate]=PRF()
        self.cate_strict[cate].true=self.cate_strict[cate].true+1
        if cate not in self.cate_relax:
            self.cate_relax[cate]=PRF()
        self.cate_relax[cate].true=self.cate_relax[cate].true+1

    def add_tp_overlap(self,cate):
        self.all_relax.true=self.all_relax.true+1
        if cate not in self.cate_relax:
            self.cate_relax[cate]=PRF()
        self.cate_relax[cate].true=self.cate_relax[cate].true+1
        # treat as false by strict
        self.all_strict.false=self.all_strict.false+1
        if cate not in self.cate_strict:
            self.cate_strict[cate]=PRF()
        self.cate_strict[cate].false=self.cate_strict[cate].false+1

    def add_nolap(self,cate):
        self.all_strict.false=self.all_strict.false+1
        self.all_relax.false=self.all_relax.false+1

        if cate not in self.cate_strict:
            self.cate_strict[cate]=PRF()
        self.cate_strict[cate].false=self.cate_strict[cate].false+1

        if cate not in self.cate_relax:
            self.cate_relax[cate]=PRF()
        self.cate_relax[cate].false=self.cate_relax[cate].false+1

    def handle(self,bio):
        llen=len(bio)

        #accumulate accuracy data
        for i in range(llen):
            if bio[i][-1].strip() == bio[i][-2].strip():
                self.acc.true=self.acc.true+1
            else:
                self.acc.false=self.acc.false+1
                
        i=0
        # handle system prediction
        while i < llen:
            if bio[i][-1] == 'o':
                i=i+1
            else:
                # find the start and end pos
                starti=i
                endi=i+1
                cate=bio[starti][-1][2:].strip()
                while endi<llen and bio[endi][-1].startswith('i-'+cate):
                    endi=endi+1
                #find the categor
                # exactly match
                if self.same(bio,starti,endi):
                    self.add_tp_strict(cate)
                # overlap        
                elif self.overlap(bio,starti,endi):
                    self.add_tp_overlap(cate)
                else: # no overlap
                    self.add_nolap(cate)
                i=endi

        #handle the ground truth label
        i=0
        while i < llen:
            if bio[i][-2] == 'o':
                i=i+1
            else:
                # find the start and end pos
                starti=i
                endi=i+1
                cate=bio[starti][-2][2:].strip()
                while endi<llen and bio[endi][-2].startswith('i-'+cate):
                    endi=endi+1
                self.gold_all=self.gold_all+1
                if cate not in self.gold_cate:
                    self.gold_cate[cate]=0
                self.gold_cate[cate]=self.gold_cate[cate]+1

                i=endi


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


def fmerge(f1, f2, ofn):
    ss1 = load_bio_file_into_sents(f1)
    ss2 = load_bio_file_into_sents(f2)

    assert len(ss1) == len(ss2), "There are {} sents in GS but {} sents in prediction".format(len(ss1), len(ss2))
    ss = []
    for s1, s2 in zip(ss1, ss2):
        assert len(s1) == len(s2), "There are {} words in GS but {} words in prediction".format(len(s1), len(s2))
        s = []
        for w1, w2 in zip(s1, s2):
            s.append((w1[0], w1[-1], w2[-1]))
        ss.append(s)

    output_bio(ss, ofn)


def main(file1, file2):
    import os
    ofn = "temp.txt"
    fmerge(file1, file2, ofn)
    evaluate = BioEval(ofn)
    evaluate.eval_fn()
    # evaluate.train_msg()
    os.remove(ofn)


def test():
    a = [['O', 'O', 'B-misc', 'O', 'O', 'B-misc', 'I-misc', 'I-misc', 'I-misc', 'I-misc', 'O']]
    b = [['O', 'O', 'B-misc', 'O', 'O', 'B-misc', 'I-misc', 'O', 'O', 'B-misc', 'O']]
    a = [[e.lower() for e in each] for each in a]
    b = [[e.lower() for e in each] for each in b]
    c = [[x for x in zip(each_a, each_b)] for each_a, each_b in zip(a, b)]
    e = BioEval(None)
    for each in c:
        e.handle(each)
    e.prf()


if __name__ == "__main__":
    import sys
    f1 = sys.argv[1]
    f2 = sys.argv[2]
    main(f1, f2)
    # test()
