# coding: utf-8

from argparse import ArgumentParser
from ast import arg
import math
import kenlm
import re

class ScoreSentence(object):
    def __init__(self, args:ArgumentParser):
        self.args = args
        self.lmean = args.length_dist_mean
        self.lvar = args.length_dist_var
        self.scale = args.length_dist_scale
        
        nd_exp = lambda x : self.scale * math.exp(-((x-self.lmean)**2)/(2*self.lvar**2))
        self.len_score = lambda l : nd_exp(l)/(math.sqrt(2*math.pi)*self.lvar)
        
        model_path = self.args.used_ngram_model
        self.ngram = kenlm.LanguageModel(model_path)
        
    def __call__(self, parsed_sentences):
        # preprocess
        parsed_sentences = re.sub('[.。．]', '。', parsed_sentences)
        parsed_sentences = re.sub('[,、，]', '、', parsed_sentences)
        parsed_sentences = re.sub('。$', '', parsed_sentences)
        parsed_sentences = re.sub('、$', '', parsed_sentences)
        parsed_sentences = re.sub('[,.、。，．][,.、。，．]+', '。', parsed_sentences)
        
        # dvision sub sentence
        parsed_sentences = re.split('。', parsed_sentences)
        parsed_sentences = [re.split('、', ps) for ps in parsed_sentences]
        
        scores = []
        sentence = []
        prob_sum = 0.0
        len_all = 0.0
        cut_flag = False
        
        # Show scores and n-gram matches
        for sentence_prd in parsed_sentences:
            for sentence_cnm in sentence_prd:
                words = ['<s>'] + sentence_cnm.split() + ['</s>']
                ngram_scores = self.ngram.full_scores(sentence_cnm)
                scores.append(ngram_scores)
                
                for i, (prob, length, oov) in enumerate(ngram_scores):
                    if prob > self.args.remove_limit and not cut_flag:
                        if words[i] != '<s>' and words[i] != '</s>':
                            sentence.append(words[i])
                            #prob_sum += prob
                            prob_sum += 1/prob
                            len_all += 1.
                    else:
                        cut_flag = True
                sentence.append('、')
            sentence.pop(-1)
            sentence.append('。')
        
        prob_mean = 0.0
        if len_all == 0:
            prob_mean = -1000.0
        else:
            #prob_mean = prob_sum / len_all
            prob_mean = len_all / prob_sum
            prob_mean += self.len_score(len_all) # add length bonus
        
        return prob_mean, sentence