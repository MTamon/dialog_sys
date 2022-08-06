from pyknp import Juman
from transformers import BertTokenizer
import re


# parameters
bert_seq_len = 512

def remove_duplication(tokens):
    """Remove duplication such as 'XYZABABABAB -> XYZAB'

    Args:
        tokens (list[str]): sentence which is tokenized

    Returns:
        list[str]: sentence which is removed duplication
    """
    if len(tokens) < 2:
        return tokens
    
    # remove duplication : 'XYZABABABAB -> XYZAB'
    tokens += ['P', 'P', 'P'] # add dummy element
    
    new_tokens = []
    for i in range(len(tokens)-3):
        source = ' '.join([tokens[i], tokens[i+1]])
        target = ' '.join([tokens[i+2], tokens[i+3]])
        new_tokens.append(tokens[i])
        
        if source == target:
            new_tokens.append(tokens[i+1])
            break
        else:
            continue
        
    tokens = new_tokens
    
    # remove duplication : 'XYZABCABCABC -> XYZABC'
    tokens += ['P', 'P', 'P', 'P', 'P']
    
    new_tokens = []
    for i in range(len(tokens)-5):
        source = ' '.join([tokens[i], tokens[i+1], tokens[i+2]])
        target = ' '.join([tokens[i+3], tokens[i+4], tokens[i+5]])
        new_tokens.append(tokens[i])
        
        if source == target:
            new_tokens.append(tokens[i+1])
            break
        else:
            continue
        
    return new_tokens

class JumanTokenizer():
    def __init__(self):
        self.juman = Juman()

    def tokenize(self, text):
        result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]

class BertWithJumanModel(BertTokenizer):
    def __init__(self, args, logger):
        super().__init__(
            args.bert_vocab,
            do_lower_case=False,
            do_basic_tokenize=False
        )
        self.juman_tokenizer = JumanTokenizer()
        self.logger = logger

    def _preprocess_text(self, text):
        return text.replace(" ", "")  # for Juman

    def get_sentence_ids(self, text):
        preprocessed_text = self._preprocess_text(text)
        tokens = self.juman_tokenizer.tokenize(preprocessed_text)
        self.logger.info(tokens)
        bert_tokens = self.tokenize(" ".join(tokens))
        ids = self.convert_tokens_to_ids(["[CLS]"] + bert_tokens[:bert_seq_len] + ["[SEP]"]) # max_seq_len-2

        recv = self.convert_ids_to_tokens(ids)

        return (ids, recv)

    def get_sentence(self, ids):
        tokens = self.convert_ids_to_tokens(ids)
        return self.convert_tokens_to_string(tokens)
    
    def div_sep(self, tokens):
        sentence = ' '.join(tokens)
        sentences = sentence.split('[SEP]')
        sentences = [sent.split() for sent in sentences]
        
        return sentences
    
    def tokens_to_sentence(self, tokens):
        sentence = self.convert_tokens_to_string(tokens)
        sentence = sentence.replace(' ','')
        sentence = re.sub('[,.、。，．][,.、。，．]+', '。', sentence)
        return sentence
