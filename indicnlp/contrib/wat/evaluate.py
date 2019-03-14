import os
import argparse
import langid
import subprocess as sp
from indicnlp.normalize.indic_normalize \
        import IndicNormalizerFactory
from indicnlp.tokenize.indic_tokenize \
        import trivial_tokenize as tokenize
import warnings


def compute_bleu(refs, hyp):
    _dir = os.path.dirname(os.path.abspath(__file__))
    moses_mbleu = os.path.join(_dir, 'multi-bleu.perl')
    refs = ' '.join(refs)
    cmd = 'perl {} {} < {}'.format(moses_mbleu, refs, hyp)
    output = sp.run(cmd, shell=True, capture_output=True).stdout
    return output.decode("utf-8")

class Evaluator:
    def __init__(self, srcs, tgt, src_lang=None):
        self.srcs = srcs
        self.tgt = tgt
        self.src_lang = src_lang
        for src in self.srcs:
            lang = self.infer_langs(src)
            if self.src_lang == None:
                self.src_lang = lang
            if not (self.src_lang == lang):
                warnings.warn("Reference languages seem to be different, please check?")

        self.tgt_lang = self.infer_langs(self.tgt)
        if not(self.tgt_lang == self.src_lang):
                warnings.warn("Hypothesis Language seem to be different, please check?")


    @staticmethod
    def add_args(parser):
        parser.add_argument('--hypothesis', type=str, required=True)
        parser.add_argument('--references', type=str, nargs='+', required=True)

    @classmethod
    def build(cls, args):
        return cls(args.references, args.hypothesis)
    

    def infer_langs(self, fname):
        first_line = next(open(fname)).strip()
        lang, logprob = langid.classify(first_line)
        return lang

    def run(self):
        tokenized_srcs = []
        for src in self.srcs:
            tokenized_file = self.normalize_and_tokenize(
                    self.src_lang, src)
            tokenized_srcs.append(tokenized_file)

        tokenized_tgt = self.normalize_and_tokenize(
                self.src_lang, self.tgt)

        bleu = compute_bleu(tokenized_srcs, tokenized_tgt)
        return {"BLEU": bleu}


    def normalize_and_tokenize(self, lang, fname):
        factory = IndicNormalizerFactory()
        normalizer = factory.get_normalizer(lang, remove_nuktas=False)
        tokenized_file = fname.replace('/', '_')
        tokenized_file = os.path.join('/tmp', tokenized_file)
        with open(fname) as istream:
            with open(tokenized_file, 'w+') as ostream:
                for line in istream:
                    line = line.strip()
                    line = normalizer.normalize(line)
                    tokens = tokenize(line, lang=self.src_lang)
                    tokenized_line = ' '.join(tokens)
                    print(tokenized_line, file=ostream)
        return tokenized_file

def main():
    parser = argparse.ArgumentParser()
    Evaluator.add_args(parser)

    args = parser.parse_args()
    evaluator = Evaluator.build(args)
    stats = evaluator.run()
    for key, val in stats.items():
        print(key, val)

if __name__ == '__main__':
    main()
