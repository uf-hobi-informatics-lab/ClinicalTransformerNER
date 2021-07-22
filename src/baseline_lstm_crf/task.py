from common_utils.bio_prf_eval import BioEval


class Task:
    def __init__(self, args):
        self.args = args

        # set up eval tool for model selection
        self.bio_eval = BioEval()
        self.bio_eval.set_logger(self.args.logger)

        # set up data loader

        # init or reload model

    def train(self):
        pass

    def _eval(self):
        pass

    def eval(self):
        pass

    def predict(self):
        pass
